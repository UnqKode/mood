const express = require("express");
const path = require("path");
const app = express();
const PORT = 10000;

app.get("/download", (req, res) => {
  const filePath = path.resolve(__dirname, "mat.m"); // MATLAB file

  res.download(filePath, "script.m", (err) => { // downloaded as script.m
    if (err) {
      console.error("Error while downloading:", err);
      if (!res.headersSent) {
        res.status(500).send("Error downloading the file");
      }
    }
  });
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}/download`);
});
