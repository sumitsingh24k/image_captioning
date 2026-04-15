const apiUrlInput = document.getElementById("apiUrl");
const modeSelect = document.getElementById("mode");
const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const generateBtn = document.getElementById("generateBtn");
const captionText = document.getElementById("captionText");

imageInput.addEventListener("change", () => {
  const [file] = imageInput.files;
  if (!file) {
    preview.style.display = "none";
    return;
  }
  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";
});

generateBtn.addEventListener("click", async () => {
  const [file] = imageInput.files;
  if (!file) {
    captionText.textContent = "Please select an image first.";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("mode", modeSelect.value);
  captionText.textContent = "Generating caption...";

  try {
    const response = await fetch(apiUrlInput.value.trim(), {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorJson = await response.json();
      throw new Error(errorJson.detail || "API request failed.");
    }

    const json = await response.json();
    captionText.textContent = json.caption || "No caption returned.";
  } catch (error) {
    captionText.textContent = `Error: ${error.message}`;
  }
});
