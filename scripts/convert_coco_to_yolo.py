import argparse
import os
from pycocotools.coco import COCO


def convert(base_dir):
    json_file = os.path.basename(base_dir) + "_with_pseudolabel.json"
    json_path = f"{base_dir}/{json_file}"
    coco = COCO(json_path)

    save_dir = f"{base_dir}/labels"
    os.makedirs(save_dir, exist_ok=True)

    for img_id in coco.getImgIds():
        img_dict = coco.loadImgs(img_id)[0]
        ann_dict_lst = coco.loadAnns(coco.getAnnIds(img_id))

        file_name = os.path.basename(img_dict["file_name"]).split('.')[0]
        W = img_dict["width"]
        H = img_dict["height"]

        with open(os.path.join(save_dir, file_name + ".txt"), "w") as f:
            for ann_dict in ann_dict_lst:
                cat_id = ann_dict['category_id']
                x, y, w, h = ann_dict['bbox']
                n_cx, n_cy = (x + w/2) / W, (y + h/2) / H
                n_w, n_h = w / W, h / H

                f.write(f"{cat_id} {n_cx} {n_cy} {n_w} {n_h}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', help="directory of which json annotation file is located")
    args = parser.parse_args()

    convert(
        base_dir=args.base_dir,
    )
