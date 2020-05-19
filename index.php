<!DOCTYPE html>
<html>
<head>
<style>

body{
	font-family:verdana;
	font-size:14px;
	background:#ffd0d0;
}

.container{
	width:1120px;
	margin:0 auto;
	border:1px solid #eeeeee;
	background:#ffffff;
	padding:10px;
}

h1{
	text-align:center;
	color:#e31616;
	font-size:20px;
	
}
table{
	border:1px solid #eeeeee;
	border-collapse: collapse;
	width:100%;
}

table th{
	border:1px solid #eeeeee;
	text-align:center;
	color:#e31616;
	height:40px;
}
table td{
	border:1px solid #eeeeee;
	padding:5px;
}

</style>
</head>
<body>





<script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.0/jquery.min.js"></script>
<script type="text/javascript">
$(document).ready(function () {
    ajax_call = function() {
        $.ajax({ //create an ajax request to load_page.php
            type: "GET",
            url: "ajax_get_data.php",
            dataType: "html", //expect html to be returned                
            success: function (response) {
             var data=$("#responsecontainer").html(response);
            }
        });
    };
    var interval = 500;
    setInterval(ajax_call, interval);
});
</script>

</script>
<div id="responsecontainer"></div>

</body>
</html>