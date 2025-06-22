import atexit
import hashlib
import shutil
import os
import random
import re
import subprocess
import tempfile
import time
from typing import Dict, Optional
from glob import glob
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from bioat.logger import LoggerManager

from openfold.config import model_config
from openfold.data import data_pipeline, feature_pipeline, templates
from openfold.utils.script_utils import load_models_from_command_line, run_model

lm = LoggerManager(mod_name="AF2indicator", log_level="DEBUG")


def hashed_random_string():
    timestamp = str(time.time()).encode("utf-8")
    hashed = hashlib.sha256(timestamp).hexdigest()
    return hashed[:12]


def strip_gaps_from_a3m(a3m_path: str, output_fasta: str):
    lm.set_names(func_name="strip_gaps_from_a3m")
    lm.logger.debug(f"get a3m_path: {a3m_path}")
    records = []
    # for rec in SeqIO.parse(a3m_path, "fasta"):
    for rec in SeqIO.parse(a3m_path, "fasta-pearson"):
        ungapped = str(rec.seq).replace("-", "").upper()
        records.append(SeqRecord(Seq(ungapped), id=rec.id, description=""))
    with open(output_fasta, "w") as f:
        SeqIO.write(records, f, "fasta")


def remove_prefix_files_and_dirs(q_fa: str):
    lm.set_names(func_name="remove_prefix_files_and_dirs")
    prefix = q_fa.replace('.fa', '')
    # 匹配所有以该 prefix 开头的路径（文件或文件夹）
    targets = glob(f"{prefix}*")
    
    for path in targets:
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
            lm.logger.debug(f"Deleted file: {path}")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            lm.logger.debug(f"Deleted directory: {path}")
    targets = glob("/dev/shm/__KMP_REGISTERED_LIB_*")
    for path in targets:
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
            
class MSAGenerator:
    def __init__(
        self,
        a3m_path: str,
        mmseqs_tmp_dir: str = "tmp",
        mmseqs_threads: int = 4,
        keep_temps: bool = False,
    ):
        lm.set_names(cls_name="MSAGenerator", func_name="__init__")
        self.session_id = f"mmseqs_session_{hashed_random_string()}"
        self.tmp_root = os.path.join(mmseqs_tmp_dir, self.session_id)
        lm.logger.debug(f"make dir: {self.tmp_root}")
        os.makedirs(self.tmp_root, exist_ok=True)

        # 保存原始无 gap 的 hits_db.fa
        self.hits_input_fasta = os.path.join(self.tmp_root, "hits_db.fa")
        lm.logger.debug(f"parse hits_db.fa from {a3m_path}")
        strip_gaps_from_a3m(a3m_path, self.hits_input_fasta)
        lm.logger.debug(f"a3m gaps were removed and save to {self.hits_input_fasta}")
        # 构建 hits_db（indexed db 用于 mmseqs search）
        self.hits_db = os.path.join(self.tmp_root, "hits_db")
        lm.logger.debug(f"use {mmseqs_threads} threads for mmseqs2 protocols")
        self.threads = mmseqs_threads
        self._build_hits_db()
        self._output_a3m_path = None

        lm.logger.debug(f"keep_temps = {keep_temps}")
        if not keep_temps:
            lm.logger.debug(
                "a cleanup protocol is register, tempfiles will be removed before the program exit."
            )
            atexit.register(self.cleanup)

    def _build_hits_db(self):
        lm.set_names(cls_name="MSAGenerator", func_name="_build_hits_db")
        if not os.path.exists(self.hits_db + ".dbtype"):
            lm.logger.debug(
                f"Building MMseqs2 DB at {self.hits_db} from {self.hits_input_fasta}..."
            )
            subprocess.run(
                ["mmseqs", "createdb", self.hits_input_fasta, self.hits_db],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            lm.logger.debug(
                f"Found existing MMseqs2 DB at {self.hits_db}. Skipping rebuild."
            )

    def _safe_rm_mmseqs_db(self, base_path):
        """删除已有的 MMseqs DB 文件（以避免 search 报错）"""
        lm.set_names(cls_name="MSAGenerator", func_name="_safe_rm_mmseqs_db")
        for ext in [".dbtype", ".index", ".lookup", ".db", ".h", ".src"]:
            path = base_path + ext
            if os.path.exists(path):
                lm.logger.debug(f"remove temp mmseqs db file {path}")
                os.remove(path)
        if self._output_a3m_path:
            lm.logger.debug(f"remove temp old a3m file {self._output_a3m_path}")
            os.remove(self._output_a3m_path)

    def cleanup(self):
        lm.set_names(cls_name="MSAGenerator", func_name="cleanup")
        if os.path.exists(self.tmp_root):
            try:
                lm.logger.debug(f"remove dir {self.tmp_root}")
                shutil.rmtree(self.tmp_root)
            except Exception as e:
                lm.logger.warning(
                    f"Warning: Failed to cleanup tmp dir {self.tmp_root}: {e}"
                )

    def generate_a3m(self, tag: str, cut_seq: str, output_a3m_path: str):
        lm.set_names(cls_name="MSAGenerator", func_name="generate_a3m")
        local_tmp = self.tmp_root

        q_fa = os.path.join(local_tmp, f"{tag}.fa")
        q_db = os.path.join(local_tmp, f"{tag}_db")
        res_db = os.path.join(local_tmp, f"{tag}_res")
        align_db = os.path.join(local_tmp, f"{tag}_align")
        align_tab = os.path.join(local_tmp, f"{tag}.tab")
        lm.logger.debug(f"use cut_seq to search, cut_seq save to {q_fa}")
        # 写入 query 序列
        SeqIO.write([SeqRecord(Seq(cut_seq), id=tag, description="")], q_fa, "fasta")

        self._safe_rm_mmseqs_db(res_db)
        # 创建 query 数据库
        lm.logger.debug(
            f"create cut_seq query db for mmseqs2: use {q_fa} to create {q_db}"
        )
        subprocess.run(
            ["mmseqs", "createdb", q_fa, q_db],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        mmseqs_tmp = q_fa.replace(".fa", "_tmps")
        # 搜索
        lm.logger.debug(f"""run mmseqs search with a very sensitive strategy: 
mmseqs search {q_db} \\
    {self.hits_db} \\
    {mmseqs_tmp} \\
    --threads {self.threads} \\
    -s 7.5 --max-seqs 1000
""")
        subprocess.run(
            [
                "mmseqs",
                "search",
                q_db,
                self.hits_db,
                res_db,
                mmseqs_tmp,
                "--threads",
                str(self.threads),
                "-s",
                "7.5",
                "--max-seqs",
                "1000",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        lm.logger.debug(f"""run mmseqs align:
mmseqs align {q_db} \\
    {self.hits_db} \\
    {res_db} \\
    {align_db} \\
    -a
""")
        # 比对并导出 cigar + tseq
        subprocess.run(
            ["mmseqs", "align", q_db, self.hits_db, res_db, align_db, "-a"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        lm.logger.debug(f"""run mmseqs convertalis:
mmseqs convertalis {q_db} \\
    {self.hits_db} \\
    {align_db} \\
    {align_tab} \\
    --format-output "target,qlen,qstart,qend,tstart,tend,tseq,cigar"
""")
        subprocess.run(
            [
                "mmseqs",
                "convertalis",
                q_db,
                self.hits_db,
                align_db,
                align_tab,
                "--format-output",
                "target,qlen,qstart,qend,tstart,tend,tseq,cigar",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        lm.logger.debug(
            f"parsing align_tab ({align_tab} and convert it to a3m format ({output_a3m_path})"
        )
        # 读取 align_tab 手动重构 A3M
        records = []
        with open(align_tab) as f:
            for line in f:
                cols = line.strip().split("\t")
                if len(cols) < 8:
                    continue
                seqid, qlen, qstart, qend, tstart, tend, tseq, cigar = cols
                qlen = int(qlen)
                qstart = int(qstart)
                qend = int(qend)
                tstart = int(tstart)

                # 计算 alignment
                alignment_seq = "-" * (qstart - 1)
                alignment_index = tstart - 1
                cigar_tuples = re.findall(r"(\d+)([MDI])", cigar)

                for count, op in cigar_tuples:
                    count = int(count)
                    if op == "M":
                        alignment_seq += tseq[
                            alignment_index : alignment_index + count
                        ].upper()
                        alignment_index += count
                    elif op == "D":
                        alignment_seq += tseq[
                            alignment_index : alignment_index + count
                        ].lower()
                        alignment_index += count
                    elif op == "I":
                        alignment_seq += "-" * count

                alignment_seq += "-" * (qlen - qend)
                records.append(SeqRecord(Seq(alignment_seq), id=seqid, description=""))

        # 添加 query 自身（无 gap）
        records.insert(0, SeqRecord(Seq(cut_seq), id=tag, description=""))
        lm.logger.debug(f"saving a3m: {output_a3m_path}")
        with open(output_a3m_path, "w") as f:
            # SeqIO.write(records, f, "fasta")
            for rec in records:
                f.write(f">{rec.id}\n{rec.seq}\n")
            # exit()
        remove_prefix_files_and_dirs(q_fa)
        return output_a3m_path


class OpenFoldPredictor:
    def __init__(
        self,
        a3m_path: str,  # WT a3m（可以从 Colabfold MSA 结果中直接取到）
        template_mmcif_dir: str | None = None,  # 结构模板, 无需指定
        config_preset: str = "model_1_ptm",
        device: str = "cuda:0",
        jax_param_path: str
        | None = "/lustre2/cqyi/hnzhao/projects/2024_ProEvo_model/openfold/openfold/resources/params/params_model_1_ptm.npz",  # "openfold/resources/params/params_model_1_ptm.npz",
        openfold_checkpoint_path: str
        | None = None,  # "openfold/resources/openfold_params/finetuning_ptm_2.pt",
        mmseqs_tmp_dir: str = "/dev/shm",  # 迭代 MSA 时文件放置在哪个文件夹
        mmseqs_threads: int = 4,
        kalign_binary_path: str = "kalign",
        keep_temps: bool = False,  # for debug
    ):
        lm.set_names(cls_name="OpenFoldPredictor", func_name="__init__")
        # ------------------------------------------------------------------->>>>>>>>>>
        # Model Config
        # ------------------------------------------------------------------->>>>>>>>>>
        self.config = model_config(
            config_preset,
            long_sequence_inference=False,
            use_deepspeed_evoformer_attention=True,  # EGFP test, False 22.48 s / True 13.82 s
        )
        # ------------------------------------------------------------------->>>>>>>>>>
        # Template and alignment processor
        # ------------------------------------------------------------------->>>>>>>>>>
        if template_mmcif_dir:
            lm.logger.debug(
                f"*.cif files in {template_mmcif_dir} will be used as template structures"
            )
            self.template_featurizer = templates.CustomHitFeaturizer(
                mmcif_dir=template_mmcif_dir,
                max_template_date="9999-12-31",
                max_hits=-1,
                kalign_binary_path=kalign_binary_path,
            )
        else:
            lm.logger.debug(
                f"template_mmcif_dir = {template_mmcif_dir}, and no template structure will be used. (faster)"
            )
            self.template_featurizer = None

        self.data_processor = data_pipeline.DataPipeline(
            template_featurizer=self.template_featurizer
        )
        # ------------------------------------------------------------------->>>>>>>>>>
        # feature processor
        # ------------------------------------------------------------------->>>>>>>>>>
        self.feature_processor = feature_pipeline.FeaturePipeline(self.config.data)
        # ------------------------------------------------------------------->>>>>>>>>>
        # load model
        # ------------------------------------------------------------------->>>>>>>>>>
        # 这里必须这么写，如果只用 jax 可以进一步简化，但是如果选用 openfold，则需要 deepseed 的缓存文件夹所以还是需要创建一个临时文件夹用
        # Load model once
        lm.logger.info(f"""load AlphaFold/OpenFold weights:
AlphaFold: {jax_param_path},
OpenFold: {openfold_checkpoint_path}
""")
        with tempfile.TemporaryDirectory() as tmp_output:
            self.model, _ = next(
                load_models_from_command_line(
                    self.config,
                    device,
                    openfold_checkpoint_path,
                    jax_param_path,
                    tmp_output,
                )
            )

        self.device = device
        self.msa_generator = MSAGenerator(
            a3m_path, mmseqs_tmp_dir, mmseqs_threads, keep_temps
        )
        self._seqtag = None
        self._a3m_path = None

    def update_msa(self, seq: str, alignment_dir: Optional[str] = None) -> str:
        lm.set_names(cls_name="OpenFoldPredictor", func_name="update_msa")
        self._seqtag = f"{hashed_random_string()}_len{len(seq)}"
        lm.logger.debug(f"MSA updating for {self._seqtag}")
        if alignment_dir is None:
            alignment_dir = os.path.join(self.msa_generator.tmp_root, "alignments")

        os.makedirs(os.path.join(alignment_dir, self._seqtag), exist_ok=True)
        a3m_path = os.path.join(alignment_dir, self._seqtag, f"{self._seqtag}.a3m")
        self.msa_generator.generate_a3m(self._seqtag, seq, a3m_path)
        return a3m_path

    def generate_feature_dict(self, seq, alignment_dir):
        lm.set_names(cls_name="OpenFoldPredictor", func_name="generate_feature_dict")
        fasta_str = f">{self._seqtag}\n{seq}\n"
        tmp_path = os.path.join(self.msa_generator.tmp_root, "tmpfa")
        os.makedirs(tmp_path, exist_ok=True)
        tmpfa = os.path.join(tmp_path, hashed_random_string() + ".fa")

        with open(tmpfa, "wt") as f:
            lm.logger.debug(
                f"writing to {tmpfa} for seqinfo: \n{fasta_str}\nto generate feature_dict"
            )
            f.write(fasta_str)

        feature_dict = self.data_processor.process_fasta(
            fasta_path=tmpfa,
            alignment_dir=os.path.join(alignment_dir, self._seqtag),
            seqemb_mode=False,
        )
        return feature_dict

    def predict(self, seq: str) -> Dict:
        lm.set_names(cls_name="OpenFoldPredictor", func_name="predict")
        alignment_dir = os.path.join(self.msa_generator.tmp_root, "alignments")
        feature_dict = self.generate_feature_dict(seq, alignment_dir)
        processed = self.feature_processor.process_features(
            feature_dict, mode="predict", is_multimer=False
        )
        processed = {
            k: torch.as_tensor(v, device=self.device) for k, v in processed.items()
        }

        with tempfile.TemporaryDirectory() as tmp_output:
            out = run_model(self.model, processed, self._seqtag, tmp_output)
        res_dict = {
            "pLDDT_resi": out["plddt"].tolist(),
            "pLDDT": float(out["plddt"].mean().item()),
            "pTM_score": out.get("ptm_score", None),
        }
        lm.logger.debug(f"predict results: {res_dict}")
        try:
            shutil.rmtree(os.path.basename(self._a3m_path))            
        except:
            pass
        return res_dict


if __name__ == "__main__":
    import random
    import time

    from Bio import SeqIO

    # 输入路径
    a3m_path = "/lustre2/cqyi/hnzhao/projects/2024_ProEvo_model/ProEvaluator/results/EGFP_20250330_meanpooling/structure/ref/ref.a3m"
    # ------------------------------------------------------------------->>>>>>>>>>
    print(f"\n--- Loading Model ---")
    start_time = time.time()
    predictor = OpenFoldPredictor(
        a3m_path=a3m_path,
        device="cuda:0",
        keep_temps=False,  # for debug
    )
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")

    # ------------------------------------------------------------------->>>>>>>>>>
    fasta_path = "/lustre2/cqyi/hnzhao/projects/2024_ProEvo_model/ProEvaluator/datasets/EGFP_20250330_meanpooling/ref.fa"  # 训练时所用的序列

    record = next(SeqIO.parse(fasta_path, "fasta"))  # 读取一次，避免循环中重复读取
    test_seq = str(record.seq)

    for i in range(5):
        # 每轮随机选择 5 个不同的位置索引
        cut_positions = sorted(random.sample(range(len(test_seq)), 5))

        # 从原始序列中去掉这些位置的字符
        test_seq = "".join(
            aa for idx, aa in enumerate(test_seq) if idx not in cut_positions
        )

        print(f"Test sequence length: {len(test_seq)}")

        start_time = time.time()
        # ✅ 显式调用 MSA 更新
        predictor.update_msa(test_seq)

        # ✅ 再调用 predict
        result = predictor.predict(test_seq)
        elapsed = time.time() - start_time

        print(f"Time taken: {elapsed:.2f} seconds")
        print("pLDDT_resi (first 5):", [round(x, 2) for x in result["pLDDT_resi"][:5]])
        print("Mean pLDDT:", round(result["pLDDT"], 2))
        print("pTM_score:", result["pTM_score"])
