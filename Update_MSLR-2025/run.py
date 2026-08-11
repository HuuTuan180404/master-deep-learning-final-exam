import argparse
from html import parser
from src import train, inference
from src.utils import setup_environment
from configs import SIConfig, USConfig  # <-- Add this import
import os
from pathlib import Path

def isExists(file_path):
    if os.path.exists(file_path):
        print("File tồn tại:", os.path.abspath(file_path))
    else:
        print("File KHÔNG tồn tại!")
        print("Đường dẫn đang kiểm tra:", os.path.abspath(file_path))


def main():
    parser = argparse.ArgumentParser(description="Sign Language Recognition Pipeline")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--infer", action="store_true", help="Run inference")
    parser.add_argument(
        "--mode",
        type=str,
        default="SI",
        choices=["SI", "US"],
        help="Select configuration mode: SI or US",
    )
    parser.add_argument(
        "--model", type=str, default="SignLanguageConformer", help="Model type"
    )
    # parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'], help='Optimizer type')
    # parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'onecycle'], help='Scheduler type')
    args = parser.parse_args()

    # Select config based on mode
    if args.mode == "SI":
        cfg = SIConfig()
    elif args.mode == "US":
        cfg = USConfig()
    else:
        raise ValueError("Invalid mode. Choose either 'SI' or 'US'.")

    cfg.MODEL_SAVE_PATH = f"outputs/models/{cfg.MODE}/{cfg.MODE}_{args.model}_best_model.pth"
    # cfg.MODEL_SAVE_PATH = f"/home/cpami-llm/CPAMI_WorkPlace/Rezwan/MSLR-2025/MSLR-Isharah/Update_MSLR-2025/outputs/models/{cfg.MODE}/{cfg.MODE}_{args.model}_checkpoint.pth"

    setup_environment(cfg)

    if args.train:
        print(
            f"Starting training process with mode: {args.mode}, model: {args.model} ..."
        )
        train(mode=args.mode, model_type=args.model)

    if args.infer:
        print(
            f"Starting inference process with mode: {args.mode}, model: {args.model} ..."
        )

        inference.generate_predictions(
            split="test", mode=args.mode, model_type=args.model
        )

        inference.generate_predictions(
            split="dev", mode=args.mode, model_type=args.model
        )


if __name__ == "__main__":
    main()
    # PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    # cfg = SIConfig()

    # print(cfg.PROJECT_ROOT)
    # print()

"""
--------------------- AdvancedSignLanguageRecognizer ---------------------
### For Mode = US (where we got high performance) # (WER) val = 55.0847  & test = 47.7756 (Winner: 2nd)

python run.py --train --mode US --model AdvancedSignLanguageRecognizer
python run.py --infer --mode US --model AdvancedSignLanguageRecognizer

--------------------- SOTA_CSLR ------------------------------------------
### For Mode = SI (where we got high performance) # (WER) val = 7.3123  & test = 13.0652 (Winner: 4th) ###

python run.py --train --mode SI --model SOTA_CSLR
python run.py --infer --mode SI --model SOTA_CSLR

"""
