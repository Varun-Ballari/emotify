var tag = document.createElement('script');
var songID = 'uhx8NjSsdY0';

tag.src = "https://www.youtube.com/iframe_api";

var firstScriptTag = document.getElementsByTagName('script')[0];
firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);

var player;
function onYouTubeIframeAPIReady() {
  player = new YT.Player('player', {
    height: $(document).height(),
    width: $(document).width(),
    videoId: songID,
    events: {
      'onReady': onPlayerReady,
      'onStateChange': onPlayerStateChange
    }
  });
}

function onPlayerReady(event) {
  event.target.playVideo();
}

var done = false;
function onPlayerStateChange(event) {
  if (event.data == YT.PlayerState.PLAYING && !done) {
    setTimeout(stopVideo, 300000);
    done = true;
  }
}

function stopVideo() {
    player.stopVideo();
}

function updateYoutubeVideo(songId) {
    player.loadVideoById(songId, 0)
}
