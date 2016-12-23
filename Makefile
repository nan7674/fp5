.PHONY: clean

MODULES = filters noise

clean:
	rm -f *.pyc *~
	$(MAKE) -C filters clean
	$(MAKE) -C noise clean