import os
import re
import tempfile
import subprocess
import uuid
import atexit
from typing import Dict, List, Optional
import torch
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from openfold.config import model_config
from openfold.data import feature_pipeline,data_pipeline, templates
from openfold.utils.script_utils import load_models_from_command_line, run_model

def strip_gaps_from_a3m(a3m_path: str, output_fasta: str):
    records = []
    # for rec in SeqIO.parse(a3m_path, "fasta"):
    for rec in SeqIO.parse(a3m_path, "fasta-pearson"):
        ungapped = str(rec.seq).replace("-", "")
        records.append(SeqRecord(Seq(ungapped), id=rec.id, description=""))
    with open(output_fasta, "w") as f:
        # tag = rec.id
        # seq = rec.seq
        # seq = seq.replace('\n', '')
        # for rec in records:
        #     f.write(f">{tag}\n{seq}\n")
        SeqIO.write(records, f, "fasta")
    with open('temp_fa.fa', "w") as f:
        SeqIO.write(records, f, "fasta")
        # tag = rec.id
        # seq = rec.seq
        # seq = seq.replace('\n', '')
        # for rec in records:
        #     f.write(f">{tag}\n{seq}\n")

class MSAGenerator:
    def __init__(self, a3m_path: str, mmseqs_tmp_dir: str = "tmp", mmseqs_threads: int = 4):
        self.session_id = f"mmseqs_session_{uuid.uuid4().hex}"
        self.tmp_root = os.path.join(mmseqs_tmp_dir, self.session_id)
        os.makedirs(self.tmp_root, exist_ok=True)

        # 保存原始无 gap 的 hits_db.fa
        self.hits_input_fasta = os.path.join(self.tmp_root, "hits_db.fa")
        strip_gaps_from_a3m(a3m_path, self.hits_input_fasta)

        # 构建 hits_db（indexed db 用于 mmseqs search）
        self.hits_db = os.path.join(self.tmp_root, "hits_db")
        self.threads = mmseqs_threads
        self._build_hits_db()

        atexit.register(self.cleanup)

    def _build_hits_db(self):
        if not os.path.exists(self.hits_db + ".dbtype"):
            print(f"[MSAGenerator] Building MMseqs2 DB at {self.hits_db} from {self.hits_input_fasta}...")
            subprocess.run(["mmseqs", "createdb", self.hits_input_fasta, self.hits_db], check=True)
        else:
            print(f"[MSAGenerator] Found existing MMseqs2 DB at {self.hits_db}. Skipping rebuild.")

    def _safe_rm_mmseqs_db(self, base_path: str):
        """删除已有的 MMseqs DB 文件（以避免 search 报错）"""
        for ext in [".dbtype", ".index", ".lookup", ".db", ".h", ".src"]:
            path = base_path + ext
            if os.path.exists(path):
                os.remove(path)
            
    def cleanup(self):
        if os.path.exists(self.tmp_root):
            try:
                import shutil
                shutil.rmtree(self.tmp_root)
            except Exception as e:
                print(f"Warning: Failed to cleanup tmp dir {self.tmp_root}: {e}")

    def generate_a3m(self, tag: str, cut_seq: str, output_a3m_path: str):
        local_tmp = self.tmp_root
    
        q_fa = os.path.join(local_tmp, f"{tag}.fa")
        q_db = os.path.join(local_tmp, f"{tag}_db")
        res_db = os.path.join(local_tmp, f"{tag}_res")
        align_db = os.path.join(local_tmp, f"{tag}_align")
        align_tab = os.path.join(local_tmp, f"{tag}.tab")
    
        # 写入 query 序列
        SeqIO.write([SeqRecord(Seq(cut_seq), id=tag, description="")], q_fa, "fasta")

        self._safe_rm_mmseqs_db(res_db)
        # 创建 query 数据库
        subprocess.run(["mmseqs", "createdb", q_fa, q_db], check=True,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
        # 搜索
        subprocess.run([
            "mmseqs", "search", q_db, self.hits_db, res_db, local_tmp,
            "--threads", str(self.threads), "-s", "7.5", "--max-seqs", "1000"
        ], check=True,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
        # 比对并导出 cigar + tseq
        subprocess.run([
            "mmseqs", "align", q_db, self.hits_db, res_db, align_db,
            "-a"
        ], check=True,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
        subprocess.run([
            "mmseqs", "convertalis", q_db, self.hits_db, align_db, align_tab,
            "--format-output", "target,qlen,qstart,qend,tstart,tend,tseq,cigar"
        ], check=True,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
        # 读取 align_tab 手动重构 A3M
        records = []
        with open(align_tab) as f:
            for line in f:
                cols = line.strip().split('\t')
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
                cigar_tuples = re.findall(r'(\d+)([MDI])', cigar)
    
                for count, op in cigar_tuples:
                    count = int(count)
                    if op == 'M':
                        alignment_seq += tseq[alignment_index:alignment_index + count].upper()
                        alignment_index += count
                    elif op == 'D':
                        alignment_seq += tseq[alignment_index:alignment_index + count].lower()
                        alignment_index += count
                    elif op == 'I':
                        alignment_seq += "-" * count
    
                alignment_seq += "-" * (qlen - qend)
                records.append(SeqRecord(Seq(alignment_seq), id=seqid, description=""))
    
        # 添加 query 自身（无 gap）
        records.insert(0, SeqRecord(Seq(cut_seq), id=tag, description=""))
    
        with open(output_a3m_path, "w") as f:
            SeqIO.write(records, f, "fasta")
        # os.system(f"cp {output_a3m_path} .")  # DEBUG TODO remove it
        return output_a3m_path
        
        
class OpenFoldPredictor:
    def __init__(
        self,
        config_preset="model_1_ptm",
        device="cuda:0",
        jax_param_path="/lustre2/cqyi/hnzhao/projects/2024_ProEvo_model/openfold/openfold/resources/params/params_model_1_ptm.npz",  # "openfold/resources/params/params_model_1_ptm.npz",
        openfold_checkpoint_path=None,  # "openfold/resources/openfold_params/finetuning_ptm_2.pt",
        template_mmcif_dir=None,  # 结构模板
        a3m_path="hits.a3m",  # WT a3m（可以从 Colabfold MSA 结果中直接取到）
        mmseqs_tmp_dir="/dev/shm",  # 迭代 MSA 时文件放置在哪个文件夹
        mmseqs_threads=4,
        kalign_binary_path="kalign"
    ):
        # ------------------------------------------------------------------->>>>>>>>>>
        # Model Config
        # ------------------------------------------------------------------->>>>>>>>>>
        self.config = model_config(
            config_preset,
            long_sequence_inference=False,
            use_deepspeed_evoformer_attention=True, # EGFP test, False 22.48 s / True 13.82 s
        )
        # ------------------------------------------------------------------->>>>>>>>>>
        # Template and alignment processor
        # ------------------------------------------------------------------->>>>>>>>>>
        # TODO 跳过模板模板对齐步骤，提升有限，明显减少 preprocessing 时间
        self.template_featurizer = templates.CustomHitFeaturizer(
            mmcif_dir=template_mmcif_dir,
            max_template_date="9999-12-31",
            max_hits=-1,
            kalign_binary_path=kalign_binary_path,
        )
        self.data_processor = data_pipeline.DataPipeline(template_featurizer=self.template_featurizer)
        # ------------------------------------------------------------------->>>>>>>>>>
        # feature processor
        # ------------------------------------------------------------------->>>>>>>>>>
        self.feature_processor = feature_pipeline.FeaturePipeline(self.config.data)
        # ------------------------------------------------------------------->>>>>>>>>>
        # load model
        # ------------------------------------------------------------------->>>>>>>>>>
        # 这里必须这么写，如果只用 jax 可以进一步简化，但是如果选用 openfold，则需要 deepseed 的缓存文件夹所以还是需要创建一个临时文件夹用
        # Load model once
        with tempfile.TemporaryDirectory() as tmp_output:
            self.model, _ = next(load_models_from_command_line(
                self.config,
                device,
                openfold_checkpoint_path,
                jax_param_path,
                tmp_output,
            ))

        self.device = device
        self.msa_generator = MSAGenerator(a3m_path, mmseqs_tmp_dir, mmseqs_threads)
        self.template_mmcif_dir = template_mmcif_dir
        
    def update_msa(self, seq: str, tag: str = "query", alignment_dir: Optional[str] = None) -> str:
        print(f"MSA updating for {tag}")
        if alignment_dir is None:
            alignment_dir = os.path.join(self.msa_generator.tmp_root, "alignments")
    
        os.makedirs(os.path.join(alignment_dir, tag), exist_ok=True)
        a3m_path = os.path.join(alignment_dir, tag, f"{tag}.a3m")
        self.msa_generator.generate_a3m(tag, seq, a3m_path)
        return a3m_path

    def generate_feature_dict(self, tag, seq, alignment_dir):
        fasta_str = f">{tag}\n{seq}"
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".fasta") as tmp:
            tmp.write(fasta_str)
            tmp_path = tmp.name
    
        try:
            feature_dict = self.data_processor.process_fasta(
                fasta_path=tmp_path,
                alignment_dir=os.path.join(alignment_dir, tag),
                seqemb_mode=False,
            )
        finally:
            os.remove(tmp_path)
        return feature_dict

    def predict(self, seq: str, tag: str = "query") -> Dict:
        alignment_dir = os.path.join(self.msa_generator.tmp_root, "alignments")
        # a3m_path = self.update_msa(seq, tag, alignment_dir) 不再调用 update_msa() 改为显式调用 MSA 更新
        feature_dict = self.generate_feature_dict(tag, seq, alignment_dir)
    
        processed = self.feature_processor.process_features(
            feature_dict, mode="predict", is_multimer=False
        )
        processed = {
            k: torch.as_tensor(v, device=self.device)
            for k, v in processed.items()
        }
    
        with tempfile.TemporaryDirectory() as tmp_output:
            out = run_model(self.model, processed, tag, tmp_output)
    
        return {
            "pLDDT_resi": out["plddt"].tolist(),
            "pLDDT": float(out["plddt"].mean().item()),
            "pTM_score": out.get("ptm_score", None)
        }

if __name__ == '__main__':
    import time
    import random
    from Bio import SeqIO

    # 输入路径
    a3m_path = "/lustre2/cqyi/hnzhao/projects/2024_ProEvo_model/ProEvaluator/results/EGFP_20250330_meanpooling/structure/ref/ref.a3m"
    template_mmcif_dir="/lustre2/cqyi/hnzhao/projects/2024_ProEvo_model/ProEvaluator/results/EGFP_20250330_meanpooling/structure/ref/ref_env/templates_101"
    # ------------------------------------------------------------------->>>>>>>>>>    
    print(f"\n--- Loading Model ---")
    start_time = time.time()
    predictor = OpenFoldPredictor(
        device="cuda:0",
        template_mmcif_dir=template_mmcif_dir,
        a3m_path=a3m_path,
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

        # test_seq = test_seq[]  # 每轮裁掉一个氨基酸
        tag = f"cut_{i}_len{len(test_seq)}"
        print(f"\n--- Prediction {i+1} ({tag}) ---")
        print(f"Test sequence length: {len(test_seq)}")
    
        start_time = time.time()
        # ✅ 显式调用 MSA 更新
        predictor.update_msa(test_seq, tag=tag)
    
        # ✅ 再调用 predict
        result = predictor.predict(test_seq, tag=tag)
        elapsed = time.time() - start_time
    
        print(f"Time taken: {elapsed:.2f} seconds")
        print("pLDDT_resi (first 5):", [round(x, 2) for x in result["pLDDT_resi"][:5]])
        print("Mean pLDDT:", round(result["pLDDT"], 2))
        print("pTM_score:", result["pTM_score"])