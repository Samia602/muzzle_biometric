"""Gradio UI to test the muzzle biometric pipeline.  Run:  python app.py"""
import gradio as gr
from PIL import ImageDraw
from src.pipeline import MuzzlePipeline
from src.registry import Registry

pipe = MuzzlePipeline()
reg = Registry()


def annotate(img, box):
    img = img.convert("RGB").copy()
    if box:
        ImageDraw.Draw(img).rectangle(box, outline="lime", width=5)
    return img


def tab_detect(img, use_det):
    if img is None: return None, None, "Upload an image."
    r = pipe.process(img, use_det)
    info = (f"Detector confidence: {r['det_conf']:.2f}\n" if r["box"] else "No box (full image used)\n")
    info += f"Embedding dim: {len(r['embedding'])}\nBiometric hash (SHA-256):\n{r['hash']}"
    return annotate(img, r["box"]), r["crop"], info


def tab_compare(a, b, thr, use_det):
    if a is None or b is None: return "Upload both images."
    ra, rb = pipe.process(a, use_det), pipe.process(b, use_det)
    s = pipe.cosine(ra["embedding"], rb["embedding"])
    verdict = "✅ SAME animal" if s >= thr else "❌ DIFFERENT animals"
    return f"{verdict}\nCosine similarity: {s:.4f}  (threshold {thr:.3f})"


def tab_register(animal_id, files, use_det):
    from PIL import Image
    if not animal_id or not files: return "Provide an ID and 1+ images."
    embs = [pipe.process(Image.open(f), use_det)["embedding"] for f in files]
    h = reg.register(animal_id.strip(), embs)
    return f"Registered '{animal_id}' with {len(embs)} image(s).\nHash: {h}\nTotal animals: {len(reg.data)}"


def tab_identify(img, thr, use_det):
    if img is None: return "Upload an image."
    if not reg.data: return "Registry empty - register animals first."
    r = pipe.process(img, use_det)
    top = reg.identify(r["embedding"])
    best_id, best_s = top[0]
    head = f"✅ MATCH: {best_id}" if best_s >= thr else "⚠️ NO MATCH (unregistered / possible fraud)"
    return head + "\n\n" + "\n".join(f"{i+1}. {k}  cos={s:.4f}" for i, (k, s) in enumerate(top))


def clear_registry():
    reg.clear(); return "Registry cleared."


with gr.Blocks(title="Muzzle Biometric Tester") as demo:
    gr.Markdown("# 🐄 Muzzle Biometric Tester\nYOLO detect → crop → ResNet+ArcFace embedding → cosine similarity")
    with gr.Row():
        thr = gr.Slider(0.0, 1.0, value=pipe.threshold, step=0.005, label="Match threshold (default = tuned on val set)")
        use_det = gr.Checkbox(value=pipe.detector is not None, label="Use YOLO muzzle detector")
    with gr.Tab("1. Detect & Hash"):
        i1 = gr.Image(type="pil", label="Cow image"); b1 = gr.Button("Run")
        with gr.Row(): o1 = gr.Image(label="Detection"); o2 = gr.Image(label="Muzzle crop")
        t1 = gr.Textbox(label="Info", lines=5)
        b1.click(tab_detect, [i1, use_det], [o1, o2, t1])
    with gr.Tab("2. Compare two images"):
        with gr.Row(): ca = gr.Image(type="pil", label="Image A"); cb = gr.Image(type="pil", label="Image B")
        bc = gr.Button("Compare"); tc = gr.Textbox(label="Result", lines=3)
        bc.click(tab_compare, [ca, cb, thr, use_det], tc)
    with gr.Tab("3. Register animal"):
        rid = gr.Textbox(label="Animal ID (e.g. COW-001)")
        rf = gr.Files(label="Muzzle images (several angles recommended)", type="filepath")
        br = gr.Button("Register"); tr = gr.Textbox(label="Result", lines=4)
        bx = gr.Button("Clear registry")
        br.click(tab_register, [rid, rf, use_det], tr); bx.click(clear_registry, None, tr)
    with gr.Tab("4. Identify / Verify"):
        ii = gr.Image(type="pil", label="Unknown animal"); bi = gr.Button("Identify")
        ti = gr.Textbox(label="Result", lines=6)
        bi.click(tab_identify, [ii, thr, use_det], ti)

if __name__ == "__main__":
    demo.launch()
