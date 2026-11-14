(() => {
  "use strict";

  const WIDTH = 1920;
  const HEIGHT = 1080;
  const PIXELS = WIDTH * HEIGHT;
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
  scratch.width = WIDTH;
  scratch.height = HEIGHT;
  const scratchContext = scratch.getContext("2d", { willReadFrequently: true });

  let sourceReal = null;
  let sourceImag = null;
  let shiftedReal = null;
  let shiftedImag = null;
  let magnitudeScale = 1;

  function createFFTPlan(length) {
    const stages = new Map();
    let stageLength = length;
    let childScratchSize = 0;
    while (stageLength > 1) {
      let radix = 2;
      while (stageLength % radix !== 0) radix += 1;
      if (radix > 5) throw new Error(`Unsupported FFT factor: ${radix}`);
      const scratchSize = 2 * stageLength + radix * childScratchSize;
      const cosine = new Float64Array(stageLength);
      const sine = new Float64Array(stageLength);
      for (let i = 0; i < stageLength; i += 1) {
        const angle = 2 * Math.PI * i / stageLength;
        cosine[i] = Math.cos(angle);
        sine[i] = Math.sin(angle);
      }
      stages.set(stageLength, { radix, subLength: stageLength / radix, scratchSize, childScratchSize, cosine, sine });
      childScratchSize = scratchSize;
      stageLength /= radix;
    }
    return { stages, scratch: new Float64Array(childScratchSize) };
  }

  function fftRecursive(inputReal, inputImag, inputOffset, inputStride, outputReal, outputImag, outputOffset, scratchOffset, length, plan, inverse) {
    if (length === 1) {
      outputReal[outputOffset] = inputReal[inputOffset];
      outputImag[outputOffset] = inputImag[inputOffset];
      return;
    }

    const stage = plan.stages.get(length);
    const { radix, subLength, childScratchSize, cosine, sine } = stage;
    for (let branch = 0; branch < radix; branch += 1) {
      fftRecursive(
        inputReal, inputImag, inputOffset + branch * inputStride, inputStride * radix,
        outputReal, outputImag, outputOffset + branch * subLength,
        scratchOffset + 2 * length + branch * childScratchSize,
        subLength, plan, inverse
      );
    }

    const sign = inverse ? 1 : -1;
    for (let frequency = 0; frequency < subLength; frequency += 1) {
      for (let branchFrequency = 0; branchFrequency < radix; branchFrequency += 1) {
        const outputIndex = frequency + subLength * branchFrequency;
        let valueReal = 0;
        let valueImag = 0;
        for (let branch = 0; branch < radix; branch += 1) {
          const inputIndex = outputOffset + branch * subLength + frequency;
          const twiddleIndex = (branch * outputIndex) % length;
          const twiddleReal = cosine[twiddleIndex];
          const twiddleImag = sign * sine[twiddleIndex];
          valueReal += outputReal[inputIndex] * twiddleReal - outputImag[inputIndex] * twiddleImag;
          valueImag += outputReal[inputIndex] * twiddleImag + outputImag[inputIndex] * twiddleReal;
        }
        plan.scratch[scratchOffset + outputIndex] = valueReal;
        plan.scratch[scratchOffset + length + outputIndex] = valueImag;
      }
    }

    for (let i = 0; i < length; i += 1) {
      outputReal[outputOffset + i] = plan.scratch[scratchOffset + i];
      outputImag[outputOffset + i] = plan.scratch[scratchOffset + length + i];
    }
  }

  function fft1d(real, imag, length, inverse, plan, workReal, workImag) {
    fftRecursive(real, imag, 0, 1, workReal, workImag, 0, 0, length, plan, inverse);
    const scale = inverse ? 1 / length : 1;
    for (let i = 0; i < length; i += 1) {
      real[i] = workReal[i] * scale;
      imag[i] = workImag[i] * scale;
    }
  }

  const fftPlans = {
    width: createFFTPlan(WIDTH),
    height: createFFTPlan(HEIGHT)
  };

  function fft2d(real, imag, inverse) {
    const lineReal = new Float64Array(WIDTH);
    const lineImag = new Float64Array(WIDTH);
    const workReal = new Float64Array(WIDTH);
    const workImag = new Float64Array(WIDTH);
    for (let y = 0; y < HEIGHT; y += 1) {
      const row = y * WIDTH;
      for (let x = 0; x < WIDTH; x += 1) {
        lineReal[x] = real[row + x];
        lineImag[x] = imag[row + x];
      }
      fft1d(lineReal, lineImag, WIDTH, inverse, fftPlans.width, workReal, workImag);
      for (let x = 0; x < WIDTH; x += 1) {
        real[row + x] = lineReal[x];
        imag[row + x] = lineImag[x];
      }
    }
    for (let x = 0; x < WIDTH; x += 1) {
      for (let y = 0; y < HEIGHT; y += 1) {
        const index = y * WIDTH + x;
        lineReal[y] = real[index];
        lineImag[y] = imag[index];
      }
      fft1d(lineReal, lineImag, HEIGHT, inverse, fftPlans.height, workReal, workImag);
      for (let y = 0; y < HEIGHT; y += 1) {
        const index = y * WIDTH + x;
        real[index] = lineReal[y];
        imag[index] = lineImag[y];
      }
    }
  }

  function centerSpectrum(real, imag) {
    const centeredReal = new Float64Array(PIXELS);
    const centeredImag = new Float64Array(PIXELS);
    const halfWidth = WIDTH >> 1;
    const halfHeight = HEIGHT >> 1;
    for (let y = 0; y < HEIGHT; y += 1) {
      for (let x = 0; x < WIDTH; x += 1) {
        const from = y * WIDTH + x;
        const to = ((y + halfHeight) % HEIGHT) * WIDTH + ((x + halfWidth) % WIDTH);
        centeredReal[to] = real[from];
        centeredImag[to] = imag[from];
      }
    }
    return [centeredReal, centeredImag];
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
    return new ImageData(pixels, WIDTH, HEIGHT);
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
    contexts.phase.putImageData(new ImageData(pixels, WIDTH, HEIGHT), 0, 0);
  }

  function applyFilter() {
    if (!shiftedReal) return;
    const radius = Number(radiusSlider.value);
    const highPass = document.querySelector('input[name="filter-mode"]:checked').value === "high";
    const filteredReal = new Float64Array(PIXELS);
    const filteredImag = new Float64Array(PIXELS);
    const pixels = new Uint8ClampedArray(PIXELS * 4);
    const halfWidth = WIDTH >> 1;
    const halfHeight = HEIGHT >> 1;
    const verticalFrequencyScale = WIDTH / HEIGHT;
    const radiusSquared = radius * radius;

    for (let y = 0; y < HEIGHT; y += 1) {
      const dy = (y - halfHeight) * verticalFrequencyScale;
      for (let x = 0; x < WIDTH; x += 1) {
        const dx = x - halfWidth;
        const index = y * WIDTH + x;
        const unshiftedIndex = ((y + halfHeight) % HEIGHT) * WIDTH + ((x + halfWidth) % WIDTH);
        const inside = dx * dx + dy * dy <= radiusSquared;
        const keep = highPass ? !inside : inside;
        if (keep) {
          filteredReal[unshiftedIndex] = shiftedReal[index];
          filteredImag[unshiftedIndex] = shiftedImag[index];
        }
        const magnitude = keep ? Math.log1p(Math.hypot(shiftedReal[index], shiftedImag[index])) / magnitudeScale * 255 : 0;
        const pixelOffset = index * 4;
        pixels[pixelOffset] = magnitude;
        pixels[pixelOffset + 1] = magnitude;
        pixels[pixelOffset + 2] = magnitude;
        pixels[pixelOffset + 3] = 255;
      }
    }

    contexts.filtered.putImageData(new ImageData(pixels, WIDTH, HEIGHT), 0, 0);
    fft2d(filteredReal, filteredImag, true);
    const reconstructedPixels = new Uint8ClampedArray(PIXELS * 4);
    for (let i = 0; i < PIXELS; i += 1) {
      const value = Math.max(0, Math.min(255, filteredReal[i]));
      const offset = i * 4;
      reconstructedPixels[offset] = value;
      reconstructedPixels[offset + 1] = value;
      reconstructedPixels[offset + 2] = value;
      reconstructedPixels[offset + 3] = 255;
    }
    contexts.reconstructed.putImageData(new ImageData(reconstructedPixels, WIDTH, HEIGHT), 0, 0);

    radiusValue.value = `${radius} px`;
    filterTag.textContent = highPass ? "HIGH-PASS" : "LOW-PASS";
    filteredCaption.textContent = highPass
      ? "Frequencies inside the radius are removed."
      : "Frequencies outside the radius are removed.";
  }

  function processImage(image) {
    scratchContext.clearRect(0, 0, WIDTH, HEIGHT);
    scratchContext.drawImage(image, 0, 0, WIDTH, HEIGHT);
    const imageData = scratchContext.getImageData(0, 0, WIDTH, HEIGHT);
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
    sourceReal = null;
    sourceImag = null;

    let maxLog = 0;
    for (let i = 0; i < PIXELS; i += 1) {
      maxLog = Math.max(maxLog, Math.log1p(Math.hypot(shiftedReal[i], shiftedImag[i])));
    }
    magnitudeScale = maxLog || 1;
    contexts.magnitude.putImageData(spectrumMagnitude(shiftedReal, shiftedImag, magnitudeScale), 0, 0);
    drawPhase(shiftedReal, shiftedImag);
    document.body.classList.add("has-image");
    status.textContent = `${image.width} × ${image.height} image transformed to 1920 × 1080.`;
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
