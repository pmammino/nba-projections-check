export default function Verdict({ text }: { text: string }) {
  const cls = text === "positive" ? "positive" : text === "negative" ? "negative" : "";
  return <span className={`pill ${cls}`}>{text}</span>;
}
