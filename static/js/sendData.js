$(document).ready(function() {
	var socket = io.connect(namespace);

	socket.on('connect', function(msg) {
		socket.emit('my event', {data: 'I\'m connected!'});
	});

	socket.on('message', function(d) {
		console.log(d)
		$('#test').html(d.emotion);
		var str = "Welcome " + d.user + "! The camera's watching you! Change your expression to change the song!";
		$('#description').html(str);
		updateYoutubeVideo(d.song)
	});
});
