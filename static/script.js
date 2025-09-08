
document.addEventListener("DOMContentLoaded", () => {
    const trainingStatus = document.getElementById("training-status");
    const synthesizingStatus = document.getElementById("synthesizing-status");
    const progressBar = document.getElementById("progress-bar");
    const logList = document.getElementById("log-list");
    const outputFile = document.getElementById("output-file");
    const samplesList = document.getElementById("samples-list");
    const uploadButton = document.getElementById("upload-button");
    const sampleFile = document.getElementById("sample-file");
    const trainButton = document.getElementById("train-button");
    const synthesizeButton = document.getElementById("synthesize-button");
    const synthesisText = document.getElementById("synthesis-text");

    const API_URL = window.location.origin;

    async function updateStatus() {
        try {
            const response = await fetch(`${API_URL}/api/status`);
            const data = await response.json();

            trainingStatus.textContent = data.training ? "Wird ausgeführt" : "Inaktiv";
            synthesizingStatus.textContent = data.synthesizing ? "Wird ausgeführt" : "Inaktiv";
            
            progressBar.innerHTML = `<div id="progress-bar-inner" style="width: ${data.progress}%"></div>`;

            logList.innerHTML = "";
            data.log.forEach(log => {
                const li = document.createElement("li");
                li.textContent = log;
                logList.appendChild(li);
            });

            if (data.output_file) {
                outputFile.innerHTML = `<a href="${API_URL}/api/output/${data.output_file}" download>${data.output_file}</a>`;
            } else {
                outputFile.textContent = "Keine";
            }

        } catch (error) {
            console.error("Fehler beim Abrufen des Status:", error);
        }
    }

    async function listSamples() {
        try {
            const response = await fetch(`${API_URL}/api/samples`);
            const data = await response.json();
            samplesList.innerHTML = "";
            data.samples.forEach(sample => {
                const li = document.createElement("li");
                li.textContent = sample;
                samplesList.appendChild(li);
            });
        } catch (error) {
            console.error("Fehler beim Abrufen der Beispiele:", error);
        }
    }

    uploadButton.addEventListener("click", async () => {
        const file = sampleFile.files[0];
        if (!file) {
            alert("Bitte wählen Sie eine Datei zum Hochladen aus.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        // Fortschrittsanzeige-Elemente
        const progressContainer = document.getElementById("upload-progress-container");
        const progressInner = document.getElementById("upload-progress-inner");
        const progressLabel = document.getElementById("upload-progress-label");
        progressContainer.style.display = "block";
        progressInner.style.width = "0%";
        progressLabel.textContent = "Upload läuft...";

        // XMLHttpRequest für Fortschritt
        const xhr = new XMLHttpRequest();
        xhr.open("POST", `${API_URL}/api/upload_sample`, true);

        xhr.upload.onprogress = function (e) {
            if (e.lengthComputable) {
                const percent = Math.round((e.loaded / e.total) * 100);
                progressInner.style.width = percent + "%";
                progressLabel.textContent = `Upload läuft... (${percent}%)`;
            }
        };
        xhr.onload = async function () {
            progressLabel.textContent = "Upload abgeschlossen!";
            setTimeout(() => {
                progressContainer.style.display = "none";
            }, 1200);
            if (xhr.status === 200) {
                const data = JSON.parse(xhr.responseText);
                if (data.error) {
                    alert(`Fehler: ${data.error}`);
                } else {
                    await listSamples();
                    await updateStatus();
                }
            } else {
                alert("Fehler beim Hochladen der Datei.");
            }
        };
        xhr.onerror = function () {
            progressLabel.textContent = "Fehler beim Upload!";
            setTimeout(() => {
                progressContainer.style.display = "none";
            }, 1200);
        };
        xhr.send(formData);
    });

    trainButton.addEventListener("click", async () => {
        try {
            const response = await fetch(`${API_URL}/api/train`, { method: "POST" });
            const data = await response.json();
            if (data.error) {
                alert(`Fehler: ${data.error}`);
            }
            await updateStatus();
        } catch (error) {
            console.error("Fehler beim Starten des Trainings:", error);
        }
    });

    synthesizeButton.addEventListener("click", async () => {
        const text = synthesisText.value;
        if (!text) {
            alert("Bitte geben Sie Text für die Synthese ein.");
            return;
        }

        const formData = new FormData();
        formData.append("text", text);

        try {
            const response = await fetch(`${API_URL}/api/synthesize`, {
                method: "POST",
                body: formData
            });
            const data = await response.json();
            if (data.error) {
                alert(`Fehler: ${data.error}`);
            }
            await updateStatus();
        } catch (error) {
            console.error("Fehler beim Starten der Synthese:", error);
        }
    });

    // Initial load
    updateStatus();
    listSamples();

    // Update status every 3 seconds
    setInterval(updateStatus, 3000);
});
