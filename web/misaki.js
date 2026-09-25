// kokoro-js phonemizes with eSpeak NG, but Paradee only ever saw phonemes from
// misaki (Kokoro's own G2P) in training. The two spell the same sounds differently
// ("aɪ" vs "I", "oʊ" vs "O", length marks, flaps), and fed raw eSpeak phonemes
// Paradee mumbles (Whisper word error rate 31% vs 3%). This is misaki's own
// eSpeak-to-misaki table (misaki/espeak.py, EspeakFallback, American English),
// adapted to eSpeak output that has no tie marks between the two halves of a
// diphthong.
const E2M = [
  ['ʔˌn̩', 'ʔn'], ['ʔn̩', 'ʔn'],
  ['aɪ', 'I'], ['aʊ', 'W'],
  ['dʒ', 'ʤ'],
  ['eɪ', 'A'], ['e', 'A'],
  ['tʃ', 'ʧ'],
  ['ɔɪ', 'Y'],
  ['ʲo', 'jo'], ['ʲə', 'jə'], ['ʲ', ''],
  ['ɚ', 'əɹ'],
  ['r', 'ɹ'],
  ['x', 'k'], ['ç', 'k'],
  ['ɬ', 'l'],
  ['̃', ''],
];

export function toMisaki(ps) {
  for (const [a, b] of E2M) ps = ps.split(a).join(b);
  ps = ps.replace(/(\S)̩/gu, 'ᵊ$1').replace(/̩/g, '');
  // misaki writes a syllabic l at the end of a word as ᵊl (naval -> nˈAvᵊl).
  // eSpeak marks it with a tie that kokoro-js drops, so key on word-final əl.
  ps = ps.replace(/əl(?=[\s;:,.!?—…"“”()]|$)/gu, 'ᵊl');
  // misaki's table maps ɐ to ə, but misaki's own lexicon keeps ɐ for the article "a"
  ps = ps.replace(/ɐ(?![\s;:,.!?—…"“”()]|$)/gu, 'ə');
  ps = ps.split('oʊ').join('O')
    .split('ɜːɹ').join('ɜɹ').split('ɜː').join('ɜɹ')
    .split('ɪə').join('iə')
    .split('ː').join('');
  ps = ps.split('o').join('ɔ');
  return ps.split('ɾ').join('T').split('ʔ').join('t');
}
