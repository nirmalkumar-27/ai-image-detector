function previewImage(event) {
    const img = document.getElementById("preview");
    img.src = URL.createObjectURL(event.target.files[0]);
    img.style.display = "block";
}

async function upload() {
    const fileInput = document.getElementById("imageInput");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select an image first");
        return;
    }

    const resultDiv = document.getElementById("result");

    // reset styles
    resultDiv.className = "result-box";
    resultDiv.innerHTML = "⏳ Processing...";

    let formData = new FormData();
    formData.append("file", file);

    try {
        let res = await fetch("https://ai-image-detector-mgab.onrender.com/detect", {
            method: "POST",
            body: formData
        });

        let data = await res.json();

        if (data.error) {
            resultDiv.innerHTML = `<p style="color:red;">Error: ${data.error}</p>`;
            return;
        }

        // color class
        if (data.label === "AI Generated") {
            resultDiv.classList.add("ai");
        } else if (data.label === "Real Image") {
            resultDiv.classList.add("real");
        } else {
            resultDiv.classList.add("uncertain");
        }

        // build result safely (NO heatmap dependency)
        let html = `
            <h2>${data.label}</h2>
            <p><b>Confidence:</b> ${data.confidence}%</p>
            <p>${data.analysis}</p>
            <p class="tech">Powered by EfficientNet-B0</p>
        `;

        resultDiv.innerHTML = html;

    } catch (err) {
        console.error(err);
        resultDiv.innerHTML = `<p style="color:red;">Request failed</p>`;
    }
}