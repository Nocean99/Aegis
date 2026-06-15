# LinkedIn Post: Aegis Vehicle Modality Benchmark

I tested Aegis Mission Intelligence on aerial vehicle detection across RGB and infrared imagery.

The interesting result was not just that the system found vehicles. It was that the best review strategy changed depending on the sensor modality.

RGB vehicle benchmark:

- Local proposal layer: 50.0% capture precision, 100.0% capture recall
- API semantic review: 73.2% capture precision, 95.3% capture recall

For RGB, API review helped. The local layer worked well as a broad first pass, then semantic review cleaned up the analyst queue without losing much recall.

Infrared vehicle benchmark:

- Local hot-blob triage: 89.4% capture precision, 100.0% capture recall
- API semantic review: 61.1% capture precision, 100.0% capture recall

For IR, local triage was stronger. The API preserved recall, but it over-kept ambiguous thermal negatives as review items.

That is an important lesson for autonomous mission systems: review policy should depend on modality.

The same mission can need different intelligence paths:

- RGB vehicle evidence benefits from selective API cleanup.
- Infrared vehicle evidence currently performs better with local hot-blob triage.
- Thermal API review needs stricter prompting or stricter NEEDS_REVIEW thresholds before it beats the local layer.

This pushes Aegis further from being a drone simulator and closer to a mission intelligence platform:

Aegis Vision Intelligence  
+ Aegis Infrared Intelligence  
+ Aegis Acoustic Intelligence

The next logical expansion is sonar/acoustic sensing.

The first acoustic milestone is not classifier training. It is making `.wav` evidence, spectrograms, acoustic segments, candidate JSON, and reports fit the same Aegis mission workflow.
