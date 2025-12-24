 ffmpeg -i vertrezmotion.ts        -vf "yadif=1"        -c:v prores_ks        -profile:v 3        -pix_fmt yuv422p10le        -vendor apl0        -color_primaries bt709 -color_trc bt709 -colorspace bt709        -c:a copy        output_test.mov

