# tf-style-transfer

Neural style transfer with TensorFlow.

Download the pretrained VGG19 weights (`imagenet-vgg-verydeep-19.mat`) from
the matconvnet site and put them in the repo root.

    pip install -r requirements.txt
    python style.py --content photo.jpg --style wave.jpg --out out/result.png

Options: `--iterations`, `--content-weight`, `--style-weight`, `--tv-weight`,
`--noise-init`. On a GTX 1060 a 512px image takes about a minute for 500
L-BFGS iterations.

Video: `sh stylize_video.sh clip.mp4 style.jpg` splits the clip into frames
with ffmpeg, stylises each frame starting from the previous result and puts
them back together. There is visible flicker; there is no temporal loss.
