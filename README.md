# PSL Isolated Recognition

Streamlit Community Cloud deployment for isolated Pakistani Sign Language recognition.

The app uses WebRTC to show the annotated live camera stream, buffers a variable-length recording after **Start Recording**, and resamples it to the model contract before prediction when **Stop and Predict** is pressed.
