let socket = new WebSocket("ws://localhost:8000/events");

socket.onmessage = function(event) {
  console.log(event.data);
  let currentSlide;
  switch(event.data){
    // mandatory gestures:
    case "right":
      console.log("received 'right' event");
      Reveal.right();
      break;
    case "left":
      console.log("received 'left' event");
      Reveal.left();
      break;
    case "rotate":
      console.log("received 'rotate' event");
      currentSlide = Reveal.getCurrentSlide();
      rotateRotatables(currentSlide, 90);  // defined in helper_methods.js
      break;
    // optional gestures:
    case "rotate_left":
      console.log("received 'rotate_left' event");
      currentSlide = Reveal.getCurrentSlide();
      rotateRotatables(currentSlide, -90);  // defined in helper_methods.js
      break;
    case "zoom_in":
      console.log("received 'zoom_in' event");
      zoom(20); // increases zoom by 20%
      break;
    case "zoom_out":
      console.log("received 'zoom_out' event");
      zoom(-20); // decreases zoom by 20%
      break;
    case "up":
      console.log("received 'up' event");
      Reveal.up();
      break;
    case "down":
      console.log("received 'down' event");
      Reveal.down();
      break;
    case "point":
      console.log("received 'point' event");
      currentSlide = Reveal.getCurrentSlide();
      togglePauseVideo(currentSlide);
      break;
    case "flip_table":
      console.log("received 'flip_table' event");
      currentSlide = Reveal.getCurrentSlide();
      rotateRotatables(currentSlide, 180);  // defined in helper_methods.js
      break;
    default:
      console.debug(`unknown message received from server: ${event.data}`);
  }
};
