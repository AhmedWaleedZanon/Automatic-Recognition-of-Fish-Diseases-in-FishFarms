 
  <script src="https://www.gstatic.com/firebasejs/7.14.0/firebase.js"></script>

<!-- TODO: Add SDKs for Firebase products that you want to use
     https://firebase.google.com/docs/web/setup#available-libraries -->
<script src="https://www.gstatic.com/firebasejs/7.14.0/firebase-analytics.js"></script>
<?php 

$host = "localhost";
$dbUser = "root";
$password = "";
$database = "gp";

$dbConn = new mysqli($host,$dbUser,$password,$database);
echo '<table>';

 echo '<tr>';

echo '<th>reading</th>';
echo  '<th>farmhardwareid</th>';

 echo '<th>date</th>';


 echo '</tr>';
if($dbConn->connect_error)
{
	die("Database Connection Error, Error No.: ".$dbConn->connect_errno." | ".$dbConn->connect_error);
}


$sql = "SELECT * FROM `reading`";
 $result=mysqli_query($dbConn,$sql);
 if (!empty($result))
 {
while($row = mysqli_fetch_array($result))
		{

$id=$row['id'];
echo$reading=$row['reading'];
echo$farmhardwareid=$row['farmhardwareid'];
echo$date=$row['date'];

    


  ?>
     
    <!-- The core Firebase JS SDK is always required and must be listed first -->

    <script>
      
        var x2= "<?php echo $reading; ?>" ;
         var x4= "<?php echo $farmhardwareid;?>" ;
        var x3= "<?php echo $date;?>";
       
         // Your web app's Firebase configuration
         
  // Your web app's Firebase configuration
   var firebaseConfig = {
    apiKey: "AIzaSyCUy6LegTtmOO3b-Q1SNLm3vbj_NpPvvio",
    authDomain: "gpfish-a9403.firebaseapp.com",
    databaseURL: "https://gpfish-a9403.firebaseio.com",
    projectId: "gpfish-a9403",
    storageBucket: "gpfish-a9403.appspot.com",
    messagingSenderId: "194235678892",
    appId: "1:194235678892:web:69e2fc9fc03b0f2a0d5f09",
    measurementId: "G-FHB87RZ6VQ"
  };
  // Initialize Firebase
  firebase.initializeApp(firebaseConfig);
 // firebase.analytics();

firebase.database().ref().child("Reading").push({
		   farmhardwareid: x4,
		   reading: x2,
		   date: x3
	  });
    </script>    
<?php

$sql = "DELETE FROM `reading` WHERE id=$id";
 $result=mysqli_query($dbConn,$sql);
    }
 }

$dbConn->close();

?>

       
