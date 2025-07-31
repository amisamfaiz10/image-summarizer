from django.shortcuts import render
from PIL import Image
from .model import processor, model, device  # ✅ Pre-loaded model, processor, and device


def index(request):
    if request.method == "POST":
        image_file = request.FILES.get("file")
        if not image_file:
            return render(request, "imager/index.html", {"error": "No file uploaded."})

        print("🖼️ File received:", image_file)
        print(f"💻 Running on: {device}")

        # --- Describe image using pre-loaded model ---
        try:
            image = Image.open(image_file).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            output_ids = model.generate(**inputs, max_new_tokens=50)
            caption = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print("📝 Caption:", caption)

            return render(request, "imager/index.html", {"caption": caption})
        
        except Exception as e:
            print("❌ Error while generating caption:", str(e))
            return render(request, "imager/index.html", {"error": "Failed to process image."})

    return render(request, "imager/index.html")


def about(request):
    return render(request, "imager/about.html")
