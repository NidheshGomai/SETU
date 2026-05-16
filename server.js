const express = require("express");
const app = express();

// Use a fixed port
const PORT = 3000;

// Test route
app.get("/", (req, res) => {
  res.send("Setu backend is running 🚀");
});

// Start server

app.listen(PORT, () => {
  console.log(`Server started successfully on http://localhost:${PORT}`);
});
/*const express = require("express");
const router = express.Router();

router.get("/", (req, res) => {
  res.send("Report route working");
});

module.exports = router;*/ //this code will connect to the main server file(server.js)
const cors = require("cors");
app.use(cors());
app.use(express.json());
"scripts"; {
  "dev"; "nodemon server.js"
}
