import PaperExplorer from "@/components/PaperExplorer";
import { getSummary } from "@/lib/data";

export const revalidate = 300;

export default async function Paper() {
  const s = await getSummary();
  return (
    <>
      <h1>Filtered paper trading</h1>
      <p className="sub">
        Every candidate bet on every captured board, filtered the way a bettor would, settled at one unit flat. The
        filtering runs in your browser on the full pool; flat stakes are what make ROI comparable across filters.
      </p>
      <PaperExplorer presets={s?.presets ?? []} />
    </>
  );
}
