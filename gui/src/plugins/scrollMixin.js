import anime from 'animejs';

export default {
    methods: {
        scroll(arg) {
            let el = document.querySelector(arg.to),
                offset = parseInt(arg.offset) || 0,
                duration = arg.duration || 3000,
                easing = arg.easing || 'easeOutExpo',
                callback = arg.callback || null;

            if ( el ) {
                anime({
                    targets: ['html', 'body'],
                    scrollTop: (el.offsetTop - offset),
                    duration: duration,
                    easing: easing,
                    complete: callback
                })
                    .finished.then(() => {
                    bus.$emit('scroll:finished', true);
                });
            }
        }
    }
}