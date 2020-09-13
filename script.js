// Each bundle exports a global object with the name of the bundle.

window.onload = () => {
  const noteSeq = {notes: [{pitch: 60, startTime: 0.0, endTime: 0.5},{pitch: 60, startTime: 0.5, endTime: 1.0},{pitch: 67, startTime: 1.0, endTime: 1.5}], totalTime: 1.5}


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
  //const modelUrl = 'https://storage.googleapis.com/magentadata/js/checkpoints/music_vae/mel_2bar_small'
  //const modelUrl = 'https://storage.googleapis.com/magentadata/js/checkpoints/music_vae/mel_4bar_small_q2'
  const modelUrl = 'https://storage.googleapis.com/magentadata/js/checkpoints/music_vae/groovae_4bar'
  const model = new music_vae.MusicVAE(modelUrl)
  model.initialize().then(() => {
    const numSamples = 3
    const randZs = tf.tidy(() => tf.randomNormal([numSamples, model.decoder.zDims]))
    document.getElementById("summaryBox").innerHTML = randZs.toString()
    const temperature = 0.5
    const stepsPerQuarter = 4
    const qpm = 120
    const outputSequences = model.decode(randZs, temperature, undefined, stepsPerQuarter, qpm)
    randZs.dispose()
    outputSequences.then((samples) => {
      samples.forEach((notes,i) => {
        //const notes = samples[0]
        //notes = core.sequences.unquantizeSequence(samples[0])

        const canv = document.createElement("canvas")
        canv.classList.add("pianoroll")
        document.getElementById("canvasHolder").appendChild(canv)

        const viz = new core.PianoRollCanvasVisualizer(notes, canv);
        const vizPlayer = new core.Player(false, {
          run: (note) => viz.redraw(note),
          stop: () => {}
        });
        let isPlaying = false
        canv.addEventListener("click",() => {
          if(isPlaying){
            vizPlayer.stop()
          }
          if(!isPlaying){
            vizPlayer.start(notes)
          }
          isPlaying = !isPlaying
        })
        //vizPlayer.start(notes)
      })
    });
  })
}
