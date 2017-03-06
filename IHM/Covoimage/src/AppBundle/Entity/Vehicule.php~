<?php
namespace AppBundle\Entity;
use Symfony\Component\Validator\Constraints as CSTRT;
use Doctrine\ORM\Mapping as ORM;
/**
 * @ORM\Entity
 * @ORM\Table(name="Vehicule")
 */
class Vehicule
{
	/**
	 * Clé primaire de Vehicule
	 * @ORM\Column(name="ID_vehicule", type="integer")
	 * @ORM\Id
	 * @CSTRT\NotBlank()
	 * @ORM\GeneratedValue(strategy="AUTO")
	 */
	private $ID_vehicule;
	/**
	 * @ORM\Column(name="modele", type="string")
	 * @CSTRT\NotBlank()
	 */
	private $modele;
	/**
	 * @ORM\Column(name="nbplaces", type="integer")
	 * @CSTRT\NotBlank()
	 * @CSTRT\Range(
     *      min = 1,
     *      max = 10,
     *      minMessage = "Un vehicule doit avoir au moins {{ limit }} place",
     *      maxMessage = "Un vehicule doit avoir au maximum {{ limit }}placesr"
     * )
	 */
	private $nbplaces;
	/**
	 * @CSTRT\NotBlank()
	 * @ORM\Column(name="description", type="string")
	 */
	private $description;

    /**
     * Get iDVehicule
     *
     * @return integer
     */
    public function getIDVehicule()
    {
        return $this->id_vehicule;
    }

    /**
     * Set modele
     *
     * @param string $modele
     *
     * @return Vehicule
     */
    public function setModele($modele)
    {
        $this->modele = $modele;

        return $this;
    }

    /**
     * Get modele
     *
     * @return string
     */
    public function getModele()
    {
        return $this->modele;
    }

    /**
     * Set nbplaces
     *
     * @param integer $nbplaces
     *
     * @return Vehicule
     */
    public function setNbplaces($nbplaces)
    {
        $this->nbplaces = $nbplaces;

        return $this;
    }

    /**
     * Get nbplaces
     *
     * @return integer
     */
    public function getNbplaces()
    {
        return $this->nbplaces;
    }

    /**
     * Set description
     *
     * @param string $description
     *
     * @return Vehicule
     */
    public function setDescription($description)
    {
        $this->description = $description;

        return $this;
    }

    /**
     * Get description
     *
     * @return string
     */
    public function getDescription()
    {
        return $this->description;
    }
}
