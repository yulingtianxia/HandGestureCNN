(function() {

  var width = 320;
  var height = 0;
  var streaming = false;

  var video = null;
  var canvas = null;
  var photo = null;
  var startbutton = null;
  var stopbutton = null;
  var output = null;
  var images = "";
  // Timer id
  var timerID = 0;

  function startup() {
    video = document.getElementById('video');
    canvas = document.getElementById('canvas');
    // photo = document.getElementById('photo');
    output = document.getElementById('output');
    startbutton = document.getElementById('startbutton');
    stopbutton = document.getElementById('stopbutton');

    navigator.getMedia = ( navigator.getUserMedia ||
                           navigator.webkitGetUserMedia ||
                           navigator.mozGetUserMedia ||
                           navigator.msGetUserMedia);

    navigator.getMedia(
      {
        video: true,
        audio: false
      },
      function(stream) {
        if (navigator.mozGetUserMedia) {
          video.mozSrcObject = stream;
        } else {
          var vendorURL = window.URL || window.webkitURL;
          video.src = vendorURL.createObjectURL(stream);
        }
        video.play();
      },
      function(err) {
        console.log("An error occured! " + err);
      }
    );

    video.addEventListener('canplay', function(ev){
      if (!streaming) {
        height = video.videoHeight / (video.videoWidth/width);

        // Firefox currently has a bug where the height can't be read from
        // the video, so we will make assumptions if this happens.

        if (isNaN(height)) {
          height = width / (4/3);
        }

        video.setAttribute('width', width);
        video.setAttribute('height', height);
        canvas.setAttribute('width', width);
        canvas.setAttribute('height', height);
        streaming = true;
      }
    }, false);

    startbutton.addEventListener('click', function(ev){
      // takepicture();
      // setTimeout(takepicture, 5000);
      startbutton.className = "hidebutton";
      stopbutton.removeAttribute("class")
      timerID = window.setInterval(takepicture, 250);
      ev.preventDefault();
    }, false);

    stopbutton.addEventListener('click', function(ev){
      // takepicture();
      // setTimeout(takepicture, 5000);
      stopbutton.className = "hidebutton";
      startbutton.removeAttribute("class");
      window.clearInterval(timerID);
      output.innerHTML += images;
      ev.preventDefault();
    }, false);

    clearphoto();
  }

  // Fill the photo with an indication that none has been
  // captured.

  function clearphoto() {
    var context = canvas.getContext('2d');
    context.fillStyle = "#AAA";
    context.fillRect(0, 0, canvas.width, canvas.height);

    var data = canvas.toDataURL('image/octet-stream');
    // photo.setAttribute('src', data);
  }

  // Capture a photo by fetching the current contents of the video
  // and drawing it into a canvas, then converting that to a PNG
  // format data URL. By drawing it on an offscreen canvas and then
  // drawing that to the screen, we can change its size and/or apply
  // other changes before drawing it.

  function takepicture() {
    var context = canvas.getContext('2d');

    if (width && height) {
      canvas.width = width;
      canvas.height = height;
      context.scale(-1,1);
      context.translate(-width, 0);
      context.drawImage(video, 0, 0, width, height);

      var data = canvas.toDataURL('image/octet-stream');
      // 下载后的问题名
      var filename = 'hand_' + (new Date()).getTime() + '.png';
      // download
      saveFile(data,filename);
      // photo.setAttribute('src', data);
      images += '<img src=' + data +'>';
    } else {
      clearphoto();
    }
  }

  // Set up our event listener to run the startup process
  // once loading is complete.
  window.addEventListener('load', startup, false);
})();

/**
 * 在本地进行文件保存
 * @param  {String} data     要保存到本地的图片数据
 * @param  {String} filename 文件名
 */
var saveFile = function(data, filename){
    var save_link = document.createElementNS('http://www.w3.org/1999/xhtml', 'a');
    save_link.href = data;
    save_link.download = filename;

    var event = document.createEvent('MouseEvents');
    event.initMouseEvent('click', true, false, window, 0, 0, 0, 0, 0, false, false, false, false, 0, null);
    save_link.dispatchEvent(event);
};
