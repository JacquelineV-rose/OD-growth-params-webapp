const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("csvfile");
const fileList = document.getElementById("file-list");
const analyzeBtn = document.getElementById("analyze-btn");

dropZone.addEventListener("click", () => fileInput.click());



dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});


dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
});


dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    fileInput.files = e.dataTransfer.files;
    updateFileList();
});



fileInput.addEventListener("change", updateFileList);


function updateFileList(){
    fileList.innerHTML = "";


    if (fileInput.files.length > 0){
        Array.from(fileInput.files).forEach(file => {
            const item = document.createElement("div");
            item.textContent = file.name;
            fileList.appendChild(item);
        });
        analyzeBtn.disabled = false;
    } else{
        analyzeBtn.disabled = true;
    }
}
