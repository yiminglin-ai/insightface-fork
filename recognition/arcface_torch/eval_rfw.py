import argparse
from utils.utils_callbacks import CallBackVerification
import torch
from backbones import get_model

bin_files = [
    "African_test",
    "Asian_test",
    "Caucasian_test",
    "Indian_test",
]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch ArcFace RFW Evaluation")

    parser.add_argument(
        "--rec_path",
        type=str,
        default="/home/yiming/data/RFW/bin_for_mxnet/RFW_test",
        help="path to rfw data",
    )
    parser.add_argument("--network", type=str, default="r50", help="backbone network")
    parser.add_argument("--weight", type=str, default="")
    parser.add_argument("--embedding_size", "-e", type=int, default=512)
    args = parser.parse_args()
    backbone = get_model(
        args.network, dropout=0.0, fp16=False, num_features=args.embedding_size
    ).cpu()
    backbone.load_state_dict(torch.load(args.weight, map_location=torch.device("cpu")))
    backbone.eval()
    if torch.cuda.is_available():
        backbone = backbone.cuda()

    verf_callback = CallBackVerification(bin_files, args.rec_path, rank=0)
    print(f"Start verification for network {args.network} weight {args.weight} on RFW")
    verf_callback(1, backbone)
