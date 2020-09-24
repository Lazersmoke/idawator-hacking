// Each bundle exports a global object with the name of the bundle.
const ZDIMS = 256

const modelUrl = 'https://storage.googleapis.com/magentadata/js/checkpoints/music_vae/mel_2bar_small'
//const modelUrl = 'https://storage.googleapis.com/magentadata/js/checkpoints/music_vae/mel_4bar_small_q2'
//const modelUrl = 'https://storage.googleapis.com/magentadata/js/checkpoints/music_vae/groovae_4bar'
const model = new music_vae.MusicVAE(modelUrl)
const modelDecode = xs => {
  //const temperature = 0.5
  const temperature = 0.0001
  const stepsPerQuarter = 4
  const qpm = 120
  return model.decode(xs, temperature, undefined, stepsPerQuarter, qpm)
}

var currentBaseZ = undefined;
var isPlaying = false

window.onload = () => {
  const controls = document.getElementById("controlArea")
  controls.style.width = "200px"
  const spreadSlider = document.createElement("input")
  const spreadDisplay = document.createElement("span")
  const spreadDisplayText = document.createTextNode("0")
  spreadDisplay.appendChild(spreadDisplayText)
  spreadSlider.type = "range"
  spreadSlider.min = "0"
  spreadSlider.max = "100"
  spreadSlider.value = "0"
  spreadSlider.disabled = true
  spreadSlider.id = "spreadSlider"
  controls.appendChild(document.createTextNode("Spread: "))
  controls.appendChild(spreadSlider)
  controls.appendChild(spreadDisplay)
  controls.appendChild(document.createElement("br"))
  spreadSlider.addEventListener("input",() => spreadDisplayText.textContent = spreadSlider.value/100)

  const regenButton = document.createElement("input")
  regenButton.value = "Regenerate"
  regenButton.type = "button"
  regenButton.disabled = true
  controls.appendChild(regenButton)

/*
 * config.json
{
  "type": "MusicVAE",
  "dataConverter": {
    "type": "MelodyConverter",
    "args": {
      "numSteps": 32,
      "minPitch": 21,
      "maxPitch": 108
    }
  }
}
*/
  model.initialize().then(() => {
    currentBaseZ = samplePrior()
    regenButton.addEventListener("click",() => regenExamples(currentBaseZ.clone()))

    modelDecode(applyPerturbations(currentBaseZ.clone(),makeRandomPerturbations(50,0.3))).then(() => {
      regenButton.disabled = false
      spreadSlider.disabled = false
    })
  })
}

function regenExamples(base){
  const numPerturb = 8
  const spreadAmount = document.getElementById("spreadSlider").value/100
  const perturbs = makeRandomPerturbations(numPerturb,spreadAmount)
  const perturbed = applyPerturbations(base,perturbs)

  document.getElementById("summaryBox").innerHTML = perturbs.toString() + "\n\n" + getSumStats(perturbs).toString()
  tfvis.render.heatmap(document.getElementById("tfvisThing"),{values: getSumStats(perturbs)})
  const baseSequence = modelDecode(base.expandDims())
  const outputSequences = modelDecode(perturbed)

  const canvHolder = document.getElementById("canvasHolder")
  while (canvHolder.firstChild) {
    canvHolder.removeChild(canvHolder.lastChild);
  }

  baseSequence.then(samples => {
    // Only one base sequence
    displaySequence(samples[0],base,"Base")
  })
  outputSequences.then(samples => {
    samples.forEach((notes,i) => {
      displaySequence(notes, perturbed.slice(i,1).squeeze(), "Perturbation #" + (i + 1))
    })
  });
}

function displaySequence(seq,zCode,name = "Unamed Note Sequence"){
  seq = core.sequences.unquantizeSequence(seq)
  const canvLabel = document.createElement("span")
  const canvLabelText = document.createTextNode(name)
  canvLabel.appendChild(canvLabelText)
  const canv = document.createElement("canvas")
  canv.classList.add("pianoroll")


  const changeButton = document.createElement("input")
  changeButton.type = "button"
  changeButton.value = "Use " + name
  changeButton.addEventListener("click",() => {
    currentBaseZ = zCode.clone()
    regenExamples(currentBaseZ)
  })
  document.getElementById("canvasHolder").appendChild(canvLabel)
  document.getElementById("canvasHolder").appendChild(changeButton)
  document.getElementById("canvasHolder").appendChild(canv)

  const viz = new core.PianoRollCanvasVisualizer(seq, canv, {noteHeight: 2, pixelsPerTimeStep: 20});
  const vizPlayer = new core.Player(false, {
    run: (note) => viz.redraw(note),
    stop: () => {isPlaying = false}
  });
  canv.addEventListener("click",() => {
    if(isPlaying){
      vizPlayer.stop()
    }
    if(!isPlaying){
      vizPlayer.start(seq)
    }
    isPlaying = !isPlaying
  })
}

function getRandZs(n){
  return tf.tidy(() => tf.randomNormal([n, ZDIMS]))
}

function samplePrior(){
  return getRandZs(1).squeeze()
}

function makeRandomPerturbations(n,spread = 0.1){
  return getRandZs(n).mul(spread)
}

function applyPerturbations(base,deltas){
  return base.expandDims().tile([deltas.shape[0],1]).add(deltas)
}

function makeRandomExamples(n,biasSize = 0,randomness = 0.1){
  const origs = getRandZs(n)
  const bias = getRandZs(1).squeeze().mul(biasSize)
  const perturbations = getRandZs(n).mul(randomness)
  return tf.stack([origs,origs.add(perturbations).add(bias)],1)
}

// examplePairs shape is [numExamples,2,ZDIMS]
function monkeySee(examplePairs){
  return tf.tidy(() => {
    const [orig, mod] = tf.split(examplePairs,2,1)
    const deltas = mod.sub(orig).squeeze()
    console.log("Deltas: ",deltas.toString())
    const sumStats = deltas.matMul(deltas.transpose())
    console.log("Sum Stats: ",sumStats.toString())
    const diag = tf.linalg.bandPart(sumStats,0,0).sum(1)
    const trace = diag.sum(0)
    diag.print()
    trace.print()
    const reducedStats = sumStats.div(trace)
    return reducedStats
  })
}

function getSumStats(deltas){
  return tf.tidy(() => {
    const sumStats = deltas.matMul(deltas.transpose())
    const diag = tf.linalg.bandPart(sumStats,0,0).sum(1)
    const trace = diag.sum(0)
    const reducedStats = sumStats.div(trace)
    return reducedStats
  })
}

