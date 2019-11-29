cd /home/zhumh/code/mmskeleton/zmh/dataset/train/video/
rm *.avi

cd /home/zhumh/code/mmskeleton/zmh/dataset/val/video/
rm *.avi

cd /home/zhumh/code/mmskeleton/zmh/dataset/test/video/
rm *.avi

ln -s /data/nturgb/nturgb+d_rgb/S00[1-9]*.avi /home/zhumh/code/mmskeleton/zmh/dataset/train/video/
ln -s /data/nturgb/nturgb+d_rgb/S01[0-1]*.avi /home/zhumh/code/mmskeleton/zmh/dataset/train/video/

ln -s /data/nturgb/nturgb+d_rgb/S01[2-4]*.avi /home/zhumh/code/mmskeleton/zmh/dataset/val/video/

ln -s /data/nturgb/nturgb+d_rgb/S01[5-6]*.avi /home/zhumh/code/mmskeleton/zmh/dataset/test/video/

