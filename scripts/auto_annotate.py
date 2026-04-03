import argparse
import os
import sys
import shutil
from pathlib import Path
from PIL import Image

def init_env():
    # Only import heavy libraries if the script runs
    # To save memory during general imports
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    return torch, Owlv2Processor, Owlv2ForObjectDetection

def convert_to_yolo_format(bbox, img_width, img_height):
    """ Converts [xmin, ymin, xmax, ymax] absolute coords to YOLO [x_center, y_center, width, height] normalized """
    xmin, ymin, xmax, ymax = bbox
    
    # Calculate center point
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    
    # Calculate width and height
    box_width = xmax - xmin
    box_height = ymax - ymin
    
    # Normalize by image dimensions
    x_center /= img_width
    y_center /= img_height
    box_width /= img_width
    box_height /= img_height
    
    # Ensure they stay between 0 and 1
    return [
        max(0.0, min(1.0, x_center)),
        max(0.0, min(1.0, y_center)),
        max(0.0, min(1.0, box_width)),
        max(0.0, min(1.0, box_height))
    ]

def auto_annotate(args):
    print("⏳ Loading Owlv2 Zero-Shot Object Detection model...")
    torch, Owlv2Processor, Owlv2ForObjectDetection = init_env()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Hardware Status: Model will be loaded on [{device.upper()}].")
    if device == "cpu":
        print("   ⚠️  WARNING: Running on CPU will be extremely slow for 5,000+ images.")
    
    model_id = "google/owlv2-base-patch16-ensemble"
    processor = Owlv2Processor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device)
    
    raw_dir = Path("data/raw_frames")
    img_dir = Path("data/train/images")
    lbl_dir = Path("data/train/labels")
    
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    if not raw_dir.exists():
        print(f"❌ Error: Source directory '{raw_dir}' does not exist.")
        sys.exit(1)
        
    all_images = sorted(list(raw_dir.glob("*.jpg")))
    if not all_images:
        print(f"❌ Error: No .jpg files found in '{raw_dir}'.")
        sys.exit(1)
        
    # Apply sampling and limits
    images_to_process = all_images[::args.sample]
    if args.limit > 0:
        images_to_process = images_to_process[:args.limit]
        
    total_imgs = len(images_to_process)
    print(f"\n🎯 Targets Configured:")
    print(f"   - Prompts: {args.prompts}")
    print(f"   - Threshold: {args.conf}")
    print(f"   - Processing: {total_imgs} images (Sample Every: {args.sample}, Limit: {args.limit})")
    print(f"   - Output Dir: data/train/\n")
    
    parsed_prompts = [p.strip() for p in args.prompts.split(",")]
    
    saved_annotations = 0
    
    model.eval()
    for idx, img_path in enumerate(images_to_process):
        percent = ((idx + 1) / total_imgs) * 100
        sys.stdout.write(f"\r⏳ Progress: {percent:.2f}% | Analysing: {img_path.name} | Total Annotated: {saved_annotations}")
        sys.stdout.flush()
        
        try:
            image = Image.open(img_path).convert("RGB")
            img_width, img_height = image.size
        except Exception as e:
            continue
            
        inputs = processor(text=[parsed_prompts], images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        target_sizes = torch.tensor([image.size[::-1]]).to(device)
        results = processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=args.conf)
        
        i = 0  # We only passed 1 image
        boxes = results[i]["boxes"]
        scores = results[i]["scores"]
        
        if len(boxes) > 0:
            # We found something! Setup YOLO txt
            label_file = lbl_dir / f"{img_path.stem}.txt"
            with open(label_file, "w", encoding="utf-8") as f:
                for box, score in zip(boxes, scores):
                    box_list = box.tolist()
                    yolo_coords = convert_to_yolo_format(box_list, img_width, img_height)
                    
                    # Target Class is always 0 for our YOLO tag class detection needs
                    class_id = 0
                    
                    line = f"{class_id} {yolo_coords[0]:.6f} {yolo_coords[1]:.6f} {yolo_coords[2]:.6f} {yolo_coords[3]:.6f}\n"
                    f.write(line)
            
            # Copy image to training folder
            shutil.copy2(img_path, img_dir / img_path.name)
            saved_annotations += 1
            
    print(f"\n\n✅ Auto-Annotation Complete!")
    print(f"🎉 Successfully annotated {saved_annotations} images out of {total_imgs}.")
    print(f"📦 Ready for YOLO training! Look inside 'data/train/' for your generated dataset.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Shot Auto-Annotator using OWLv2")
    parser.add_argument("--prompts", type=str, default="price tag, product label", help="Comma-separated text prompts to search for (default: 'price tag, product label')")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold (default: 0.2)")
    parser.add_argument("--limit", type=int, default=0, help="Limit total images to process (0 = Process All)")
    parser.add_argument("--sample", type=int, default=1, help="Process every Nth image (default: 1)")
    
    args = parser.parse_args()
    auto_annotate(args)
