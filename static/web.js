window.onload = function () {

  // Definitions
  var canvas = document.getElementById("paint-canvas");
  var context = canvas.getContext("2d");
  var boundings = canvas.getBoundingClientRect();

  // Specifications
  var mouseX = 0;
  var mouseY = 0;
  context.strokeStyle = 'black'; // initial brush color
  context.lineWidth = 50; // initial brush width
  context.lineCap="round";

  var isDrawing = false;


  // Handle Colors
  var colors = document.getElementsByClassName('colors')[0];
  colors.childNodes.forEach(function(child, i){
      console.log(child.tagName)
      if(child.tagName === "BUTTON"){
          child.style.backgroundColor = child.value
      }

  });

  colors.addEventListener('click', function(event) {
    context.strokeStyle = event.target.value || 'black';
  });

  // Handle Brushes
  var brushes = document.getElementsByClassName('brushes')[0];

  brushes.addEventListener('click', function(event) {
    context.lineWidth = event.target.value || 1;
  });

  // Mouse Down Event
  canvas.addEventListener('mousedown', function(event) {
    isDrawing = true;
    context.save();
    mouseX=canvas.pageX-this.offsetLeft;
    mouseY=canvas.pageY-this.offsetTop
  });

  // Mouse Move Event
  canvas.addEventListener('mousemove', function(event) {
    if(isDrawing){
      context.beginPath();
      context.moveTo(event.clientX-boundings.left,event.clientY-boundings.top);
      context.lineTo(mouseX,mouseY);
      context.stroke();
      context.closePath();
      mouseX=event.clientX-boundings.left;
      mouseY=event.clientY-boundings.top
    }
  });

  // Mouse Up Event
  canvas.addEventListener('mouseup', function(event) {
    isDrawing = false;
  });

  // Handle Clear Button
  var clearButton = document.getElementById('clear');

  clearButton.addEventListener('click', function() {
    context.clearRect(0, 0, canvas.width, canvas.height);
  });

  // Handle Save Button
  // var saveButton = document.getElementById('save');

  // saveButton.addEventListener('click', function() {
  //   var imageName = prompt('Please enter image name');
  //   var canvasDataURL = canvas.toDataURL();
  //   var a = document.createElement('a');
  //   a.href = canvasDataURL;
  //   a.download = imageName || 'drawing';
  //   a.click();
  // });

  var generateButton = document.getElementById('generate');

  generateButton.addEventListener('click', function() {
    var canvasDataURL = canvas.toDataURL();
    fetch("/generate", {
        method: "POST",
        body: canvasDataURL
    }).then(function(res){
        return res.json()
    }).then(function(data){
        var body = document.getElementsByTagName("BODY")[0];
        var img = document.createElement("img")
        img.src = data.url
        document.body.appendChild(img)
    });
  });
};

