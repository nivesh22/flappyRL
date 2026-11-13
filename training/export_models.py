"""
Exports trained PyTorch checkpoints to ONNX (.onnx) for ONNX Runtime Web
and packages neural network layer weights + training metrics into a unified JSON bundle
for 60 FPS browser inference and live hidden-layer activation visualization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import onnx
import onnxruntime as ort
import numpy as np
import torch

from agents.dqn import DQNAgent, ARCHITECTURES
from agents.double_dqn import DoubleDQNAgent
from agents.dueling_dqn import DuelingDQNAgent
from agents.ppo import PPOAgent


ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
ONNX_DIR = CHECKPOINTS_DIR / "onnx"
FRONTEND_MODELS_DIR = ROOT_DIR / "frontend" / "public" / "models"
FRONTEND_DATA_DIR = ROOT_DIR / "frontend" / "src" / "data"


def export_module_to_onnx(module: torch.nn.Module, onnx_path: Path) -> None:
    module.eval()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.zeros(1, 5, dtype=torch.float32)

    torch.onnx.export(
        module,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["state"],
        output_names=["output"],
        dynamic_axes={"state": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,
    )

    # Ensure all tensor data is embedded inside the single .onnx file (no external .data file)
    model_proto = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save_model(model_proto, str(onnx_path), save_as_external_data=False)
    ext_data = onnx_path.with_suffix(".onnx.data")
    if ext_data.exists():
        ext_data.unlink()

    # Verify with ONNX Runtime
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"state": dummy_input.numpy()})[0]
    with torch.no_grad():
        torch_out = module(dummy_input).numpy()
    np.testing.assert_allclose(ort_out, torch_out, rtol=1e-4, atol=1e-5)


def export_all_models() -> Dict[str, Any]:
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    FRONTEND_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)

    bundle: Dict[str, Any] = {
        "networks": {},
        "time_machine": {},
        "q_learning": {},
    }

    # 1. Export Tiny, Simple, Deep, Double, Dueling DQN, and PPO to ONNX + JSON weights
    nn_configs = [
        ("tiny_dqn", DQNAgent(hidden_dims=ARCHITECTURES["tiny"]), "tiny_dqn.pt"),
        ("simple_dqn", DQNAgent(hidden_dims=ARCHITECTURES["small"]), "simple_dqn.pt"),
        ("deep_dqn", DQNAgent(hidden_dims=ARCHITECTURES["deep"]), "deep_dqn.pt"),
        ("double_dqn", DoubleDQNAgent(hidden_dims=ARCHITECTURES["medium"]), "double_dqn.pt"),
        ("dueling_dqn", DuelingDQNAgent(hidden_dims=ARCHITECTURES["medium"]), "dueling_dqn.pt"),
    ]

    for model_id, agent, ckpt_name in nn_configs:
        agent.load(CHECKPOINTS_DIR / ckpt_name)
        onnx_file = ONNX_DIR / f"{model_id}.onnx"
        export_module_to_onnx(agent.online_net, onnx_file)
        # Copy to frontend/public/models/ for ONNX Runtime Web
        (FRONTEND_MODELS_DIR / f"{model_id}.onnx").write_bytes(onnx_file.read_bytes())
        bundle["networks"][model_id] = agent.online_net.export_weights_dict()
        bundle["networks"][model_id]["onnx_url"] = f"/models/{model_id}.onnx"
        print(f"[ONNX] Exported & verified {model_id}.onnx")

    # Export PPO
    ppo_agent = PPOAgent(hidden_dims=ARCHITECTURES["medium"])
    ppo_agent.load(CHECKPOINTS_DIR / "ppo.pt")
    ppo_onnx = ONNX_DIR / "ppo.onnx"
    export_module_to_onnx(ppo_agent.net, ppo_onnx)
    (FRONTEND_MODELS_DIR / "ppo.onnx").write_bytes(ppo_onnx.read_bytes())
    bundle["networks"]["ppo"] = ppo_agent.net.export_weights_dict()
    bundle["networks"]["ppo"]["onnx_url"] = "/models/ppo.onnx"
    print("[ONNX] Exported & verified ppo.onnx")

    # 2. Export Training Time Machine Checkpoints
    for stage_id in ["ep0", "ep100", "ep350", "ep700", "ep1200"]:
        tm_agent = DoubleDQNAgent(hidden_dims=ARCHITECTURES["medium"])
        tm_agent.load(CHECKPOINTS_DIR / f"time_machine_{stage_id}.pt")
        onnx_file = ONNX_DIR / f"time_machine_{stage_id}.onnx"
        export_module_to_onnx(tm_agent.online_net, onnx_file)
        (FRONTEND_MODELS_DIR / f"time_machine_{stage_id}.onnx").write_bytes(
            onnx_file.read_bytes()
        )
        bundle["time_machine"][stage_id] = tm_agent.online_net.export_weights_dict()
        bundle["time_machine"][stage_id]["onnx_url"] = (
            f"/models/time_machine_{stage_id}.onnx"
        )
        print(f"[ONNX] Exported & verified time_machine_{stage_id}.onnx")

    # 3. Include Tabular Q-Learning table and training metrics
    q_path = CHECKPOINTS_DIR / "q_learning.json"
    if q_path.exists():
        bundle["q_learning"] = json.loads(q_path.read_text(encoding="utf-8"))

    metrics_path = ROOT_DIR / "experiments" / "training_metrics.json"
    if metrics_path.exists():
        bundle["experiments"] = json.loads(metrics_path.read_text(encoding="utf-8"))

    out_json = FRONTEND_DATA_DIR / "modelsBundle.json"
    out_json.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"[Export] Wrote frontend bundle to {out_json}")
    return bundle


if __name__ == "__main__":
    export_all_models()
