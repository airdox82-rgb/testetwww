
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

        try {
            const response = await fetch(`${API_URL}/api/upload_sample`, {
                method: "POST",
                body: formData
            });
            const data = await response.json();
            if (data.error) {
                alert(`Fehler: ${data.error}`);
            } else {
                await listSamples();
                await updateStatus();
            }
        } catch (error) {
            console.error("Fehler beim Hochladen der Datei:", error);
        }
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
