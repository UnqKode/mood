const express = require("express");
const path = require("path");
const app = express();
const PORT = 10000;

// Helper function
function downloadFile(res, id) {
  const filePath = path.resolve(__dirname, `${id}.m`); // Direct .m file

  res.download(filePath, `${id}.m`, (err) => {
    if (err) {
      console.error(`Error downloading ${id}.m`, err);
      if (!res.headersSent) {
        res.status(500).send("Error downloading the file");
      }
    }
  });
}

// -------- SEPARATE ENDPOINTS --------
app.get("/m1", (req, res) => downloadFile(res, "m11"));
app.get("/m2", (req, res) => downloadFile(res, "m22"));
app.get("/m3", (req, res) => downloadFile(res, "m33"));
app.get("/m4", (req, res) => downloadFile(res, "m44"));
app.get("/m5", (req, res) => downloadFile(res, "m55"));
app.get("/m6", (req, res) => downloadFile(res, "m66"));
app.get("/m7", (req, res) => downloadFile(res, "m77"));
app.get("/m8", (req, res) => downloadFile(res, "m88"));
app.get("/m9", (req, res) => downloadFile(res, "m99"));

app.listen(PORT, () => {
  console.log(`Server running on:`);
});
