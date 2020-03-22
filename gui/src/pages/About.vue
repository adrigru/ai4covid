<template>
    <div>
        <div class="page-header clear-filter">
            <parallax
                    class="page-header-image"
                    style="background:linear-gradient(#8e8e8e 0%, #ffffff 80%)">
            </parallax>
            <div class="container">
            </div>
            <div class="content-center brand">
                <h1 class="h1-seo">AI 4 Covid</h1>
                <h3>AI powered solution for diagnosing COVID-19 cases based on x-ray scans.</h3>
                <n-button type="primary" @click="scroll2()" size="lg">See more</n-button>
            </div>
        </div>
        <div class="main">
            <div class="section section-images">
                <div class="container">
                    <div class="row">
                        <div class="col-md-12">
                            <div class="content-center brand">

                            <h1 class="h1-seo;text-align:center">AI Powered solution to boost </h1>
                            </div>
                            <p>I'm baby fashion axe jean shorts typewriter craft beer, swag ugh locavore PBR&B you probably haven't heard of them listicle.
                                Next level banjo sustainable, tofu ugh blog woke post-ironic activated charcoal mumblecore gluten-free DIY fixie man braid.
                                Four dollar toast tattooed bespoke ramps pickled authentic you probably haven't heard of them normcore.
                                Beard raw denim kitsch listicle, humblebrag single-origin coffee hot chicken hoodie pop-up deep v pug.

                                Quinoa blue bottle artisan, jianbing umami 90's seitan.
                                Brunch health goth 8-bit yr deep v banjo truffaut swag small batch leggings raw denim chia actually neutra lyft.
                                Semiotics viral pop-up flannel dreamcatcher woke 90's distillery cray jianbing butcher.
                                Asymmetrical mustache yuccie, beard prism +1 williamsburg air plant tattooed tumeric poutine af slow-carb unicorn.
                                Drinking vinegar cliche salvia etsy enamel pin.
                                Aesthetic semiotics beard church-key gochujang, occupy sriracha gastropub swag chartreuse tilde pop-up hell of tote bag.
                                Typewriter deep v selvage, hexagon everyday carry knausgaard asymmetrical umami flannel intelligentsia mustache pitchfork mumblecore keffiyeh cred. </p>

                            </div>

                        </div>
                    </div>
                </div>
            </div>
        </div>
</template>
<script>
    import {Parallax, Card} from '@/components';
    import Typography from './components/Typography';
    import Button from '../components/Button';
    import scrollTo from '../plugins/scrollMixin';

    export default {
        name: 'index',
        mixins: [scrollTo],
        bodyClass: 'index-page',
        components: {
            [Button.name]: Button,
            Parallax,
            Typography,
            Card,
        },
        methods: {
            scroll2() {
                this.scroll({
                    to: '.main'
                });
            }

        }

    };
</script>
<style></style>
