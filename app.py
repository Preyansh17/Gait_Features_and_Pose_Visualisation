from flask import *  
app = Flask(__name__)  
import os
import script
import cv2
app.config['UPLOAD_FOLDER'] = '/var/www/html/videos/'  #(CHANGE)This is the directory where you want to save the videos which are recorded on the frontend
@app.route('/', methods = ['POST','GET'])  
def index():  
    return render_template("index.html")  
 
@app.route('/success', methods = ['POST','GET'])  
def success():  
    if request.method == 'POST':  
            video_data = request.data
            with open('/var/www/html/videos/video.mp4', 'wb') as f:  
                f.write(video_data)

            script.final()
            response =  "Video available for download"
            
            return response
    return render_template('success.html')
            
@app.route('/download/',methods = ['GET'])
def download():
    if request.method == 'GET':    
        #path = "html2pdf.pdf"
        #path = "info.xlsx"
        path = "/root/fin.mp4"    #(CHANGE) This is the destination of the output video to be downloaded. It should be same as the path given in script.py while generating the output video
        #path = "sample.txt"
        return send_file(path, as_attachment=True)
  

if __name__ == '__main__':  
    app.run(port = '8080',debug = True)  
