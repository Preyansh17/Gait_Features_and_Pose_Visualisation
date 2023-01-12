from flask import *  
app = Flask(__name__)  
import os
import script
import cv2
app.config['UPLOAD_FOLDER'] = '/var/www/html/videos/'
@app.route('/', methods = ['POST','GET'])  
def index():  
    return render_template("index.html")  
 
@app.route('/success', methods = ['POST','GET'])  
def success():  
    if request.method == 'POST':  
            video_data = request.data
            # video_data.save('/var/www/html/videos/vid2.avi')
            with open('/var/www/html/videos/video.mp4', 'wb') as f:
                f.write(video_data)

            script.final()
            response =  "Video available for download"
            
            return response
    return render_template('success.html')

camera = cv2.VideoCapture('/root/fin.mp4')
def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='video/mp4; boundary=frame')
            
@app.route('/download/',methods = ['GET'])
def download():
    if request.method == 'GET':    
        #path = "html2pdf.pdf"
        #path = "info.xlsx"
        path = "/root/fin.mp4"
        #path = "sample.txt"
        return send_file(path, as_attachment=True)
  

if __name__ == '__main__':  
    app.run(port = '8080',debug = True)  
