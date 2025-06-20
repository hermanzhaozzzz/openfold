import os
import tempfile
import torch
import numpy as np
from openfold.config import model_config
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.utils.script_utils import load_models_from_command_line, run_model

class OpenFoldPredictor:
    def __init__(
        self,
        config_preset="model_1_ptm",
        model_device="cuda:0",
        jax_param_path=None,  # "openfold/resources/params/params_model_1_ptm.npz",
        openfold_checkpoint_path=None,  # "openfold/resources/openfold_params/finetuning_ptm_2.pt",
        template_mmcif_dir=None,
        kalign_binary_path="kalign"
    ):
        # Config
        self.config = model_config(
            config_preset,
            long_sequence_inference=False,
            # use_deepspeed_evoformer_attention=False,
            # Time taken: 22.48 seconds
            use_deepspeed_evoformer_attention=True
            # Time taken: 13.82 seconds
        )
        # self.config.data.common.use_templates = False  # 跳过模板模板对齐步骤，明显减少 preprocessing 时间

        # Template and alignment processor
        self.template_featurizer = templates.CustomHitFeaturizer(
            mmcif_dir=template_mmcif_dir,
            max_template_date="9999-12-31",
            max_hits=-1,
            kalign_binary_path=kalign_binary_path,
        )
        self.data_processor = data_pipeline.DataPipeline(
            template_featurizer=self.template_featurizer
        )
        self.feature_processor = feature_pipeline.FeaturePipeline(self.config.data)

        # Load model once
        with tempfile.TemporaryDirectory() as tmp_output:
            self.model, _ = next(load_models_from_command_line(
                self.config,
                model_device,
                openfold_checkpoint_path,
                jax_param_path,
                tmp_output,
            ))

        self.device = model_device

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

    def predict(self, tag: str, seq: str, alignment_dir: str) -> dict:
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
            "plddt": out["plddt"],  # Tensor[n]
            "ptm_score": out["ptm_score"],  # float
        }


if __name__ == '__main__':
    import time
    from Bio import SeqIO

    
    # ------------------------------------------------------------------->>>>>>>>>>    
    print(f"\n--- Loading Model ---")
    start_time = time.time()
    predictor = OpenFoldPredictor(
        config_preset="model_1_ptm",
        model_device="cuda:0",
        jax_param_path="openfold/resources/params/params_model_1_ptm.npz",
        # openfold_checkpoint_path="openfold/resources/openfold_params/finetuning_ptm_2.pt",
        template_mmcif_dir="/lustre2/cqyi/hnzhao/projects/2024_ProEvo_model/zhn_test/test_openfold/template_mmcif_dir",
    )
    elapsed = time.time() - start_time
    print(f"Time taken: {elapsed:.2f} seconds")
    # ------------------------------------------------------------------->>>>>>>>>>
    fasta_path = "/lustre2/cqyi/hnzhao/projects/2024_ProEvo_model/zhn_test/test_openfold/input_fasta_dir/egfp.fa"

    for i in range(3):
        record = next(SeqIO.parse(fasta_path, "fasta"))
        print(f"\n--- Prediction {i+1} ---")
        start_time = time.time()
        result = predictor.predict(
            record.id, 
            str(record.seq), 
            alignment_dir="/lustre2/cqyi/hnzhao/projects/2024_ProEvo_model/zhn_test/test_openfold/alignments"
        )
        elapsed = time.time() - start_time
        print(f"Time taken: {elapsed:.2f} seconds")
        # print("pLDDT:", result["plddt"][:5].tolist(), "...")  # 简略输出前5个
        print("pLDDT:", [round(i, 2) for i in result["plddt"].tolist()], "...")  # 简略输出前5个
        print("pLDDT:", result["plddt"].mean())
        print("ptm_score:", result["ptm_score"])