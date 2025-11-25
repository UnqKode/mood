const express = require("express");
const path = require("path");
const fs = require("fs");
const app = express();
const PORT = 10000;

app.get("/download", (req, res) => {
  const txtFilePath = path.resolve(__dirname, "mat.txt");
  const mFilePath = path.resolve(__dirname, "mat.m");

  try {
    // Read content from .txt file
    const content = fs.readFileSync(txtFilePath, 'utf8');
    
    // Write content to .m file
    fs.writeFileSync(mFilePath, content, 'utf8');
    
    console.log("✓ Converted mat.txt to mat.m");
    
    // Download the .m file
    res.download(mFilePath, "mat.m", (err) => {
      if (err) {
        console.error("Error while downloading:", err);
        if (!res.headersSent) {
          res.status(500).send("Error downloading the file");
        }
      }
    });
    
  } catch (error) {
    console.error("Error during conversion:", error);
    res.status(500).send("Error converting or downloading file");
  }
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}/download`);
});