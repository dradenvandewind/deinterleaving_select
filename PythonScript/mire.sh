ffmpeg -i 2009_Interviews.mkv -i 2009_Interviews_x264_esdif.mkv -i 2009_Interviews_x264_bwdif.mkv -i 2009_Interviews_nnedi.mkv \
-filter_complex "\
[0:v]scale=720:406,drawtext=text='initial':fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=10[v0]; \
[1:v]scale=720:406,drawtext=text='esdif':fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=10[v1]; \
[2:v]scale=720:406,drawtext=text='bwdif':fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=10[v2]; \
[3:v]scale=720:406,drawtext=text='3meedi':fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=10[v3]; \
[v0][v1]hstack[top]; \
[v2][v3]hstack[bottom]; \
[top][bottom]vstack[out]" \
-map "[out]" -c:v libx264 -preset medium -crf 23 mire.mkv
