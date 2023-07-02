# Functions for TS anomaly detection scripts


def score(ts_train, ts_test, scorer, scaler = None):
  
  # Scale series if applicable
  if scaler:
    ts_train = scaler.fit_transform(ts_train)
    ts_test = scaler.transform(ts_test)
  
  # Train scorer
  _ = scorer.fit(ts_train)
  
  # Score series
  scores_train = scorer.score(ts_train)
  scores_test = scorer.score(ts_test)
  scores = scores_train.append(scores_test)
  
  return scores_train, scores_test, scores


def detect(scores_train, scores_test, detector):
  
  # Train & detect
  anoms_train = detector.fit_detect(scores_train)
  anoms_test = detector.detect(scores_test)
  anoms = anoms_train.append(anoms_test)
  
  return anoms_train, anoms_test, anoms
