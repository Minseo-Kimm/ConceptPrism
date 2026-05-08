# Demo concepts

Each subdirectory holds a few reference images for one concept. `init_prompts.json` maps each concept name to a list of VLM-generated captions (one per reference image, sorted-filename order) — these initialize ConceptPrism's residual tokens at the start of Stage 1.

## Adding a new concept

1. Drop reference images into `datasets/<concept_name>/`. JPEG / JPG / PNG only.
2. Generate residual-token init captions:

   ```bash
   export GEMINI_API_KEY=...   # https://ai.google.dev
   python scripts/generate_init_prompts.py \
       --concept_name <concept_name> \
       --dataset_root datasets \
       --init_prompts_path datasets/init_prompts.json
   ```

   This appends `"<concept_name>": [caption_for_image_0, ...]` to `init_prompts.json`. The list length must equal the number of reference images; the script enforces that.

3. Train as usual:

   ```bash
   bash scripts/train_sd21.sh <concept_name>
   ```

## Filename ordering

Reference images are sorted by filename and assigned residual-token slots in that order (`<|res0|>` for the first image, `<|res1|>` for the second, etc.). Stage 1 and Stage 2 use the same sort, so per-image residual tokens stay aligned across stages. Don't rename files between training and inference unless you're prepared to retrain.

## Image attribution

The reference images bundled here are redistributed under their original release terms; see the linked sources for full licenses.

- **DreamBooth dataset** ([Ruiz et al., CVPR 2023](https://github.com/google/dreambooth)) — 30 subjects: `backpack`, `backpack_dog`, `bear_plushie`, `berry_bowl`, `can`, `candle`, `cat`, `cat2`, `clock`, `colorful_sneaker`, `dog`, `dog2`, `dog3`, `dog5`, `dog6`, `dog7`, `dog8`, `duck_toy`, `fancy_boot`, `grey_sloth_plushie`, `monster_toy`, `pink_sunglasses`, `poop_emoji`, `rc_car`, `red_cartoon`, `robot_toy`, `shiny_sneaker`, `teapot`, `vase`, `wolf_plushie`.
- **CustomConcept101** ([Kumari et al., CVPR 2023](https://github.com/adobe-research/custom-diffusion#getting-started)) — 14 subjects: `actionfigure`, `book`, `car8`, `cat3`, `cat7`, `chair`, `dog9`, `doll_plushie`, `gnome_toy`, `guitar`, `happy_plushie`, `lobster_plushie`, `pink_plushie`, `purse`.
- **Freepik** (https://www.freepik.com) — 9 abstract styles: `chinese`, `cyberpunk`, `impressionism`, `pointillism`, `sunset`, `swirls`, `vectorjuice`, `watercolor`, `yosa`.

