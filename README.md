# Master_UTCN_AC_Genetic_Algorithms_for_Optimizing_the_Scheduling_of_Electrical_Devices_Python
Soluția propusă utilizează un algoritm genetic de optimizare pentru rezolvarea 
problemei de planificare a funcționării dispozitivelor electrice. Fiecare individ 
din populație este reprezentat sub forma unei matrice binare, în care 
dispozitivele sunt prezentate pe rânduri, iar orele zilei pe coloane. O valoare 
de 1 în matrice indică funcționarea dispozitivului în intervalul orar respectiv, 
iar o valoare de 0 indică inactivitatea acestuia. Calitatea soluțiilor este evaluată 
prin intermediul funcției de fitness, care integrează costul energiei electrice, 
confortul utilizatorului și raportul PAR (Peak-to-Average Ratio). Procesul de 
optimizare este alcătuit din inițializarea populației, evaluarea funcției de 
fitness, aplicarea mecanismelor de elitism și selecție, aplicarea operatorilor 
genetici de încrucișare și mutație și implementarea constrângerilor specifice 
fiecărui tip de dispozitiv. Procesul evolutiv se repetă până la atingerea 
numărului maxim de generații stabilit.
