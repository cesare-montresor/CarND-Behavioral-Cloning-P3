from moviepy.editor import ImageSequenceClip
import argparse
import glob

default_video = './videos/20170420-164131/'

def main():
    parser = argparse.ArgumentParser(description='Create driving video.')
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default=default_video,
        help='Path to image folder. The video will be created from these images.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='FPS (Frames per second) setting for the video.')
    args = parser.parse_args()

    video_file = args.image_folder + 'video.mp4'
    print("Creating video {}, FPS={}".format(video_file, args.fps))
    images = glob.glob(args.image_folder+'*')
    clip = ImageSequenceClip(images, fps=args.fps)
    clip.write_videofile(video_file)


if __name__ == '__main__':
    main()
