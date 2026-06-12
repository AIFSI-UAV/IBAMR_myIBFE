function [St, f_tail, A_pp] = compute_St(filename, U_ref, t_start)
%COMPUTE_ST Estimate tail-beat Strouhal number from fish_diagnostics.csv.
%
% St = f_tail*A_pp/U_ref, where A_pp is the peak-to-peak tail excursion.
%
% Example:
%   [St, f_tail, A_pp] = compute_St("fish_diagnostics.csv", 0.4, 2.0);

  if nargin < 1 || isempty(filename)
    filename = "fish_diagnostics.csv";
  end
  if nargin < 2
    U_ref = NaN;
  end
  if nargin < 3
    t_start = 0.0;
  end

  data = readtable(filename);
  required = ["time", "tail_lateral"];
  if ~all(ismember(required, string(data.Properties.VariableNames)))
    error("The diagnostics file must contain time and tail_lateral columns.");
  end

  keep = isfinite(data.time) & isfinite(data.tail_lateral) & data.time >= t_start;
  time = data.time(keep);
  tail = data.tail_lateral(keep);
  if numel(time) < 8
    error("At least eight diagnostic samples are required after t_start.");
  end

  dt = median(diff(time));
  if ~(isfinite(dt) && dt > 0.0)
    error("Diagnostic times must be strictly increasing.");
  end

  tail = tail - mean(tail);
  A_pp = max(tail) - min(tail);
  spectrum = abs(fft(tail));
  frequencies = (0:numel(tail)-1)'/(numel(tail)*dt);
  positive = 2:floor(numel(tail)/2) + 1;
  [~, peak_index] = max(spectrum(positive));
  f_tail = frequencies(positive(peak_index));

  if isfinite(U_ref) && U_ref > 0.0
    St = f_tail*A_pp/U_ref;
  else
    St = NaN;
    warning("U_ref must be positive to evaluate St; returning NaN.");
  end
end
