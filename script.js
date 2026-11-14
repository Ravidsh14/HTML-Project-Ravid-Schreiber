(() => {
  "use strict";

  const SIZE = 256;
  const PIXELS = SIZE * SIZE;
  const input = document.getElementById("image-input");
  const dropZone = document.getElementById("drop-zone");
  const status = document.getElementById("status");
  const radiusSlider = document.getElementById("radius-slider");
  const radiusValue = document.getElementById("radius-value");
  const filterTag = document.getElementById("filter-tag");
  const filteredCaption = document.getElementById("filtered-caption");
  const canvases = {
    original: document.getElementById("original-canvas"),
    magnitude: document.getElementById("magnitude-canvas"),
    phase: document.getElementById("phase-canvas"),
    filtered: document.getElementById("filtered-canvas"),
    reconstructed: document.getElementById("reconstructed-canvas")
  };
  const contexts = Object.fromEntries(
    Object.entries(canvases).map(([name, canvas]) => [name, canvas.getContext("2d", { willReadFrequently: true })])
  );
  const scratch = document.createElement("canvas");
  scratch.width = SIZE;
  scratch.height = SIZE;
  const scratchContext = scratch.getContext("2d", { willReadFrequently: true });

  let sourceReal = null;
  let sourceImag = null;
  let shiftedReal = null;
  let shiftedImag = null;
  let magnitudeScale = 1;

  function fft1d(real, imag, inverse) {
    const n = real.length;
    for (let i = 1, j = 0; i < n; i += 1) {
      let bit = n >> 1;
      while (j & bit) {
        j ^= bit;
        bit >>= 1;
      }
      j ^= bit;
      if (i < j) {
        [real[i], real[j]] = [real[j], real[i]];
        [imag[i], imag[j]] = [imag[j], imag[i]];
      }
    }

    for (let length = 2; length <= n; length *= 2) {
      const angle = (inverse ? 2 : -2) * Math.PI / length;
      const stepReal = Math.cos(angle);
      const stepImag = Math.sin(angle);
      const half = length >> 1;
      for (let start = 0; start < n; start += length) {
        let twiddleReal = 1;
        let twiddleImag = 0;
        for (let offset = 0; offset < half; offset += 1) {
          const even = start + offset;
          const odd = even + half;
          const oddReal = real[odd] * twiddleReal - imag[odd] * twiddleImag;
          const oddImag = real[odd] * twiddleImag + imag[odd] * twiddleReal;
          const evenReal = real[even];
          const evenImag = imag[even];
          real[even] = evenReal + oddReal;
          imag[even] = evenImag + oddImag;
          real[odd] = evenReal - oddReal;
          imag[odd] = evenImag - oddImag;

          const nextReal = twiddleReal * stepReal - twiddleImag * stepImag;
          twiddleImag = twiddleReal * stepImag + twiddleImag * stepReal;
          twiddleReal = nextReal;
        }
      }
    }

    if (inverse) {
      for (let i = 0; i < n; i += 1) {
        real[i] /= n;
        imag[i] /= n;
      }
    }
  }

  function fft2d(real, imag, inverse) {
    const lineReal = new Float64Array(SIZE);
    const lineImag = new Float64Array(SIZE);
    for (let y = 0; y < SIZE; y += 1) {
      const row = y * SIZE;
      for (let x = 0; x < SIZE; x += 1) {
        lineReal[x] = real[row + x];
        lineImag[x] = imag[row + x];
      }
      fft1d(lineReal, lineImag, inverse);
      for (let x = 0; x < SIZE; x += 1) {
        real[row + x] = lineReal[x];
        imag[row + x] = lineImag[x];
      }
    }
    for (let x = 0; x < SIZE; x += 1) {
      for (let y = 0; y < SIZE; y += 1) {
        const index = y * SIZE + x;
        lineReal[y] = real[index];
        lineImag[y] = imag[index];
      }
      fft1d(lineReal, lineImag, inverse);
      for (let y = 0; y < SIZE; y += 1) {
        const index = y * SIZE + x;
        real[index] = lineReal[y];
        imag[index] = lineImag[y];
      }
    }
  }

  function centerSpectrum(real, imag) {
    const centeredReal = new Float64Array(PIXELS);
    const centeredImag = new Float64Array(PIXELS);
    const half = SIZE >> 1;
    for (let y = 0; y < SIZE; y += 1) {
      for (let x = 0; x < SIZE; x += 1) {
        const from = y * SIZE + x;
        const to = ((y + half) % SIZE) * SIZE + ((x + half) % SIZE);
        centeredReal[to] = real[from];
        centeredImag[to] = imag[from];
      }
    }
    return [centeredReal, centeredImag];
  }

  function uncenterSpectrum(real, imag) {
    return centerSpectrum(real, imag);
  }

  function spectrumMagnitude(real, imag, scale) {
    const pixels = new Uint8ClampedArray(PIXELS * 4);
    for (let i = 0; i < PIXELS; i += 1) {
      const value = Math.log1p(Math.hypot(real[i], imag[i])) / scale * 255;
      const offset = i * 4;
      pixels[offset] = value;
      pixels[offset + 1] = value;
      pixels[offset + 2] = value;
      pixels[offset + 3] = 255;
    }
    return new ImageData(pixels, SIZE, SIZE);
  }

  function drawPhase(real, imag) {
    const pixels = new Uint8ClampedArray(PIXELS * 4);
    for (let i = 0; i < PIXELS; i += 1) {
      const value = (Math.atan2(imag[i], real[i]) + Math.PI) / (2 * Math.PI) * 255;
      const offset = i * 4;
      pixels[offset] = value;
      pixels[offset + 1] = value;
      pixels[offset + 2] = value;
      pixels[offset + 3] = 255;
    }
    contexts.phase.putImageData(new ImageData(pixels, SIZE, SIZE), 0, 0);
  }

  function applyFilter() {
    if (!shiftedReal) return;
    const radius = Number(radiusSlider.value);
    const radiusSquared = radius * radius;
    const highPass = document.querySelector('input[name="filter-mode"]:checked').value === "high";
    const filteredReal = new Float64Array(PIXELS);
    const filteredImag = new Float64Array(PIXELS);
    const filteredCanvasReal = new Float64Array(PIXELS);
    const filteredCanvasImag = new Float64Array(PIXELS);
    const center = SIZE >> 1;

    for (let y = 0; y < SIZE; y += 1) {
      const dy = y - center;
      for (let x = 0; x < SIZE; x += 1) {
        const dx = x - center;
        const index = y * SIZE + x;
        const inside = dx * dx + dy * dy <= radiusSquared;
        const keep = highPass ? !inside : inside;
        if (keep) {
          filteredReal[index] = shiftedReal[index];
          filteredImag[index] = shiftedImag[index];
          filteredCanvasReal[index] = shiftedReal[index];
          filteredCanvasImag[index] = shiftedImag[index];
        }
      }
    }

    contexts.filtered.putImageData(spectrumMagnitude(filteredCanvasReal, filteredCanvasImag, magnitudeScale), 0, 0);
    const [unshiftedReal, unshiftedImag] = uncenterSpectrum(filteredReal, filteredImag);
    fft2d(unshiftedReal, unshiftedImag, true);
    const pixels = new Uint8ClampedArray(PIXELS * 4);
    for (let i = 0; i < PIXELS; i += 1) {
      const value = Math.max(0, Math.min(255, unshiftedReal[i]));
      const offset = i * 4;
      pixels[offset] = value;
      pixels[offset + 1] = value;
      pixels[offset + 2] = value;
      pixels[offset + 3] = 255;
    }
    contexts.reconstructed.putImageData(new ImageData(pixels, SIZE, SIZE), 0, 0);

    radiusValue.value = `${radius} px`;
    filterTag.textContent = highPass ? "HIGH-PASS" : "LOW-PASS";
    filteredCaption.textContent = highPass
      ? "Frequencies inside the radius are removed."
      : "Frequencies outside the radius are removed.";
  }

  function processImage(image) {
    scratchContext.clearRect(0, 0, SIZE, SIZE);
    scratchContext.drawImage(image, 0, 0, SIZE, SIZE);
    const imageData = scratchContext.getImageData(0, 0, SIZE, SIZE);
    sourceReal = new Float64Array(PIXELS);
    sourceImag = new Float64Array(PIXELS);
    for (let i = 0; i < PIXELS; i += 1) {
      const offset = i * 4;
      sourceReal[i] = 0.299 * imageData.data[offset] + 0.587 * imageData.data[offset + 1] + 0.114 * imageData.data[offset + 2];
      imageData.data[offset] = sourceReal[i];
      imageData.data[offset + 1] = sourceReal[i];
      imageData.data[offset + 2] = sourceReal[i];
      imageData.data[offset + 3] = 255;
    }
    contexts.original.putImageData(imageData, 0, 0);
    sourceImag.fill(0);
    fft2d(sourceReal, sourceImag, false);
    [shiftedReal, shiftedImag] = centerSpectrum(sourceReal, sourceImag);

    let maxLog = 0;
    for (let i = 0; i < PIXELS; i += 1) {
      maxLog = Math.max(maxLog, Math.log1p(Math.hypot(shiftedReal[i], shiftedImag[i])));
    }
    magnitudeScale = maxLog || 1;
    contexts.magnitude.putImageData(spectrumMagnitude(shiftedReal, shiftedImag, magnitudeScale), 0, 0);
    drawPhase(shiftedReal, shiftedImag);
    document.body.classList.add("has-image");
    status.textContent = `${image.width} × ${image.height} image transformed to 256 × 256.`;
    status.classList.remove("error");
    applyFilter();
  }

  function loadFile(file) {
    if (!file || !file.type.startsWith("image/")) {
      status.textContent = "Please choose a valid image file.";
      status.classList.add("error");
      return;
    }
    const image = new Image();
    const objectUrl = URL.createObjectURL(file);
    image.onload = () => {
      try {
        processImage(image);
      } catch (error) {
        status.textContent = `Could not process this image: ${error.message}`;
        status.classList.add("error");
      } finally {
        URL.revokeObjectURL(objectUrl);
      }
    };
    image.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      status.textContent = "The selected file could not be decoded as an image.";
      status.classList.add("error");
    };
    image.src = objectUrl;
  }

  input.addEventListener("change", () => loadFile(input.files[0]));
  radiusSlider.addEventListener("input", applyFilter);
  document.querySelectorAll('input[name="filter-mode"]').forEach((radio) => {
    radio.addEventListener("change", applyFilter);
  });
  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.add("is-dragging");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.remove("is-dragging");
    });
  });
  dropZone.addEventListener("drop", (event) => loadFile(event.dataTransfer.files[0]));
})();
