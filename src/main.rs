use opencv::{
    core::{Rect, Size, Vector},
    imgproc, objdetect, prelude::*, videoio,
};

fn main() {
    
    //opencv::core::set_num_threads(0).unwrap(); // Disable OpenCV multi-threading
    // Testvideo Resolution:2560x1440 FPS:60 Length:58seconds
    /* Allow auto Multithreading by opencv: 
    real    9m27.286s    //Used real clock time
    user    109m39.050s  //Used CPU time
    sys     1m7.785s     
    */
    /*
    Dont allow Multithreading by opencv: (1 Thread)
    real    68m28.241s
    user    71m16.683s
    sys     0m15.844s 
    */

    let cascade_path = "haarcascade_frontalface_default.xml"; // Path to Haar cascade file
    let input_video_path = "testvid2k.mp4"; // Path to input MP4 video
    let output_video_path = "output_result.mp4"; // Path to save the processed video

    // Load Haar cascade
    let mut face_cascade = objdetect::CascadeClassifier::new(cascade_path)
        .expect("Failed to load Haar cascade file. Ensure the path is correct and the file exists.");

    // Open the input video file
    let mut cap = videoio::VideoCapture::from_file(input_video_path, videoio::CAP_ANY)
        .expect("Failed to open input video file");
    if !cap.is_opened().unwrap() {
        panic!("Unable to open input video file. Check the path and file format.");
    }

    // Get video properties
    let frame_width = cap.get(videoio::CAP_PROP_FRAME_WIDTH).unwrap() as i32;
    let frame_height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT).unwrap() as i32;
    let fps = cap.get(videoio::CAP_PROP_FPS).unwrap() as f64;
    let codec = videoio::VideoWriter::fourcc('m', 'p', '4', 'v').unwrap();

    // Create VideoWriter to save output
    let mut writer = videoio::VideoWriter::new(
        output_video_path,
        codec,
        fps,
        Size::new(frame_width, frame_height),
        true, // Is color
    )
    .expect("Failed to create VideoWriter");
    if !writer.is_opened().unwrap() {
        panic!("Unable to open VideoWriter for the output file.");
    }

    // Process frames and write to the output file
    let mut frame = Mat::default();
    while cap.read(&mut frame).unwrap() {
        if frame.empty() {
            break;
        }

        // Convert to grayscale for face detection
        let mut gray_frame = Mat::default();
        imgproc::cvt_color(&frame, &mut gray_frame, imgproc::COLOR_BGR2GRAY, 0).unwrap();

        // Detect faces
        let mut faces = Vector::<Rect>::new();
        face_cascade
            .detect_multi_scale(
                &gray_frame,
                &mut faces,
                1.05,
                6,
                0,
                Size::new(50, 50),
                Size::new(0, 0),
            )
            .unwrap();

        // Draw rectangles around detected faces
        for face in faces {
            imgproc::rectangle(
                &mut frame,
                face,
                opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0), // Color: Green
                2,                                               // Thickness
                imgproc::LINE_8,
                0,
            )
            .unwrap();
        }

        // Write the processed frame to the output file
        writer.write(&frame).expect("Failed to write frame to output file");
    }

    println!("Processing complete. The output video is saved at: {}", output_video_path);
}
