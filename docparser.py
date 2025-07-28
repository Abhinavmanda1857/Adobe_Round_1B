import os
import json
from pdf2image import convert_from_path
from PIL import Image
import torch
import pytesseract
import numpy as np
import cv2
from doclayout_yolo import YOLOv10


class DocParser:
    def __init__(
        self,
        pdfs_dir="./pdfs",
        outputs_dir="./outputs",
        model_path="models/last.pt",
        imgsz=896,
        conf=0.2,
        batch_size=8,
        tess_lang='eng'
    ):
        self.pdfs_dir = pdfs_dir
        self.outputs_dir = outputs_dir
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self.batch_size = batch_size
        self.tess_lang = tess_lang

        os.makedirs(self.outputs_dir, exist_ok=True)
        torch.set_num_threads(8)
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model = YOLOv10(self.model_path)

    @staticmethod
    def pil_to_cv2(pil_img):
        np_img = np.array(pil_img)
        return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR) if np_img.ndim == 3 else np_img

    @staticmethod
    def extract_ocr_text(cv2image, bbox, lang):
        x1, y1, x2, y2 = map(int, bbox)
        roi = cv2image[y1:y2, x1:x2]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        text = pytesseract.image_to_string(roi_rgb, lang=lang)
        return text.strip()

    @staticmethod
    def batch_list(lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i:i + batch_size]

    def detections_to_json(self, images_cv2, det_results, results_accum, start_page_no=1):
        for idx, (cv2_img, result) in enumerate(zip(images_cv2, det_results)):
            page_no = start_page_no + idx - 1
            for det in result.boxes.data:
                x1, y1, x2, y2, conf, cls = det.tolist()
                text = self.extract_ocr_text(cv2_img, (x1, y1, x2, y2), self.tess_lang)
                results_accum.append({
                    "page": page_no,
                    "class": int(cls),
                    "text": text,
                    "y1": y1,
                    "y2": y2
                })

    @staticmethod
    def assign_header_levels(results):
        header_indices = [i for i, r in enumerate(results) if r["class"] == 7]
        heights = np.array([results[i]["y2"] - results[i]["y1"] for i in header_indices])

        if len(heights) == 0:
            for r in results:
                r["level"] = ""
            return results
        elif len(heights) == 1:
            results[header_indices[0]]["level"] = "H1"
        elif len(heights) == 2:
            hi = np.argsort(-heights)
            results[header_indices[hi[0]]]["level"] = "H1"
            results[header_indices[hi[1]]]["level"] = "H2"
        else:
            h1_thres = np.percentile(heights, 66)
            h2_thres = np.percentile(heights, 33)
            for idx, h in zip(header_indices, heights):
                if h >= h1_thres:
                    results[idx]["level"] = "H1"
                elif h >= h2_thres:
                    results[idx]["level"] = "H2"
                else:
                    results[idx]["level"] = "H3"
        for r in results:
            if r["class"] != 7:
                r["level"] = ""
        return results

    def process_pdf(self, pdf_file):
        pdf_path = os.path.join(self.pdfs_dir, pdf_file)
        output_json_path = os.path.join(self.outputs_dir, os.path.splitext(pdf_file)[0] + ".json")
        print(f"Processing {pdf_path}...")

        pages = convert_from_path(pdf_path, dpi=96)
        pil_pages = [page.resize((self.imgsz, self.imgsz), Image.LANCZOS) for page in pages]
        all_cv2_images = [self.pil_to_cv2(page) for page in pil_pages]

        results_to_save = []
        for batch_start, batch_cv2_images in enumerate(self.batch_list(all_cv2_images, self.batch_size)):
            det_results = self.model.predict(batch_cv2_images, imgsz=self.imgsz, conf=self.conf, device=self.device)
            self.detections_to_json(
                batch_cv2_images, det_results, results_to_save,
                start_page_no=batch_start * self.batch_size + 1
            )

        sorted_results = sorted(results_to_save, key=lambda x: (x["page"], x["y1"]))

        # Title logic
        doc_title = ""
        for r in sorted_results:
            if r["class"] == 10 and r["page"] == 0:
                doc_title = r["text"]
            elif r["class"] == 10:
                r["class"] = 7

        sorted_results = self.assign_header_levels(sorted_results)

        for r in sorted_results:
            r.pop("y1", None)
            r.pop("y2", None)

        outline = [
            {"level": r["level"], "text": r["text"], "page": r["page"]}
            for r in sorted_results if r.get("level", "").startswith("H")
        ]

        output_json = {
            "title": doc_title,
            "outline": outline
        }

        with open(output_json_path, 'w', encoding='utf-8') as jf:
            json.dump(output_json, jf, ensure_ascii=False, indent=4)

    def process_all_pdfs(self):
        pdf_files = [f for f in os.listdir(self.pdfs_dir) if f.lower().endswith(".pdf")]
        if not pdf_files:
            print(f"No PDFs found in {self.pdfs_dir}")
            return
        for pdf_file in pdf_files:
            self.process_pdf(pdf_file)
