# CHROMA KEYING
This simple programs lets you remove the background from a video, and substitute it by an image of your choice. Use the preview window to select which colour is going to be taken as background and the softnes of the foreground edges.

## Dependencies:
* [Python](https://www.python.org/doc/) - 3.10.5
* [OpenCV](https://docs.opencv.org/4.6.0/) - 4.6.0
* [Numpy](https://numpy.org/doc/stable/) - 1.22.4

## How to use:
1. Put the *.mp4* video and the background image inside the */data* folder.
2. Go to the */src* folder and execute the main program:
```console
    $ python ChromaKeying.py
```
3. If everything went fine, a new window will pop up showing the preview menu, just like this:

![alt text](https://github.com/Josgonmar/Chroma-keying/blob/master/docs/preview.JPG?raw=true)

Click on the background and then use both the tolerance and softness trackbars to get the desired output. After that, just press *ESC* and the video will be processed. It will appear as *video_output.mp4* inside the */data* folder.

![alt text](https://github.com/Josgonmar/Chroma-keying/blob/master/docs/output.JPG?raw=true)

## License:
Feel free to use this programa whatever you like!