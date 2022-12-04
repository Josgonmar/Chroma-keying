import cv2
import numpy as np
import os

class GreenScreener():
    __background_img = None
    __video_cap = None
    __video_writer = None
    __window_name = 'Configure the parameters to get the best results and press ESC when done.'
    __selected_colour = np.zeros(3, dtype=np.uint32)
    __lower_bound = np.zeros(3, dtype=np.uint32)
    __upper_bound = np.zeros(3, dtype=np.uint32)
    __kernel_size = 1

    def __init__(self):
        self.__getStuff()
        cv2.namedWindow(self.__window_name, cv2.WINDOW_AUTOSIZE)

    def run(self):
        # Get the first video frame to show results
        has_frame, frame = self.__video_cap.read()
        if not has_frame:
            cv2.destroyAllWindows()
            pass
        
        # With the first frame data, create the video writer
        frame_w   = int(self.__video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h   = int(self.__video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_fps = int(self.__video_cap.get(cv2.CAP_PROP_FPS))
        fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
        file_out_mp4 = '../data/video_out.mp4'
        self.__video_writer  = cv2.VideoWriter(file_out_mp4, fourcc_mp4, frame_fps, (frame_w,frame_h))

        cv2.setMouseCallback(self.__window_name, self.__mouseCB, frame)
        cv2.createTrackbar('Tolerance', self.__window_name, 1, 100, self.__toleranceCB)
        cv2.createTrackbar('Softness', self.__window_name, 0, 10, self.__softnessCB)

        # Reduce the size of the first frame, only for previewing purposes:
        width = int(frame.shape[1] / 2)
        height = int(frame.shape[0] / 2)
        dim = (width, height)
        resized_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

        # Making sure the background image has the same size as the video frame
        self.__background_img = cv2.resize(self.__background_img, (frame.shape[1], frame.shape[0]), interpolation = cv2.INTER_AREA)

        # Configure the chroma paramteres then, press 'esc'
        k = 0
        while k != 27:
            mask = cv2.inRange(resized_frame, self.__lower_bound, self.__upper_bound)
            mask_inv = cv2.bitwise_not(mask)
            masked_frame = cv2.bitwise_and(resized_frame, resized_frame, mask=mask_inv)
            medianBlurred = cv2.medianBlur(masked_frame, self.__kernel_size)

            cv2.imshow(self.__window_name, medianBlurred)
            k = cv2.waitKey(20)

        print('Processing video. This may take a while...')

        new_frame = self.__processFrame(frame)
        self.__video_writer.write(new_frame)

        # After saving the first processed frame, we do the same for the exact same thing for the rest of the video frames
        while self.__video_cap.isOpened():
            has_frame, frame = self.__video_cap.read()
            if not has_frame:
                break
            new_frame = self.__processFrame(frame)
            self.__video_writer.write(new_frame)

        cv2.destroyAllWindows()
        self.__video_cap.release()
        self.__video_writer.release()
    
    def __processFrame(self, frame):
        mask = cv2.inRange(frame, self.__lower_bound, self.__upper_bound)
        mask_inv = cv2.bitwise_not(mask)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask_inv)
        medianBlurred = cv2.medianBlur(masked_frame, self.__kernel_size)
        bkg_mask = cv2.bitwise_and(self.__background_img, self.__background_img, mask=mask)
        final_frame = cv2.bitwise_or(bkg_mask, medianBlurred)

        return final_frame

    # Callback functions
    def __mouseCB(self, action, x, y, flags, data):
        if action == cv2.EVENT_LBUTTONDOWN:
            self.__selected_colour = data[y,x,:]

    def __toleranceCB(self, *args):
        self.__lower_bound = np.round(self.__selected_colour * (1 - args[0]/100)) - 10
        self.__lower_bound[self.__lower_bound<0] = 0

        self.__upper_bound = np.round(self.__selected_colour * (1 + args[0]/100)) + 10
        self.__upper_bound[self.__upper_bound>255] = 255

    def __softnessCB(self, *args):
        if (args[0] % 2 == 0):
            self.__kernel_size = args[0] + 1
        else:
            self.__kernel_size = args[0]

    def __getStuff(self):
        try:
            for file in os.listdir('../data'):
                if file.endswith('.mp4'):
                    self.__video_cap = cv2.VideoCapture('../data/' + file)
                else:
                    self.__background_img = cv2.imread('../data/' + file, cv2.IMREAD_COLOR)
        except:
            print('[ERROR] An error occured while getting the images')


if __name__ == "__main__":
    obj = GreenScreener()
    obj.run()