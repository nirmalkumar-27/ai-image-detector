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

        if (data.label === "AI Generated") {
            resultDiv.classList.add("ai");
        } else if (data.label === "Real Image") {
            resultDiv.classList.add("real");
        } else {
            resultDiv.classList.add("uncertain");
        }

        resultDiv.innerHTML = `
            <h2>${data.label}</h2>
            <p><b>Confidence:</b> ${data.confidence}%</p>
            <p>${data.analysis}</p>
            <h3>Model Focus</h3>
            <img src="data:image/jpeg;base64,${data.heatmap}" width="250"/>
            <p class="tech">Powered by EfficientNet-B0 + Explainable AI</p>
        `;

    } catch (err) {
        resultDiv.innerHTML = `<p style="color:red;">Request failed</p>`;
    }
}