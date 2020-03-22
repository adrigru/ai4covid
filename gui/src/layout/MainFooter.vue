<template>
    <footer
            class="footer"
            :class="{ [`footer-${type}`]: type }"
            :data-background-color="backgroundColor"
    >
        <div class="container">
            <nav>
                <ul>
                    <li>
                        <a href="https://github.com/adeaeede/ai4covid">Github </a>
                    </li>
                </ul>
            </nav>
            <div class="copyright">
                &copy; {{ year }} Adrian Gruszczynski, Kuba Weimann
            </div>
        </div>
    </footer>
</template>
<script>
    export default {
        props: {
            backgroundColor: String,
            type: String
        },
        data() {
            return {
                year: new Date().getFullYear()
            };
        }
    };
</script>
<style></style>
